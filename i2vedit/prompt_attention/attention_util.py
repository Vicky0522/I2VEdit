"""
Collect all function in prompt_attention folder.
Provide a API `make_controller' to return an initialized AttentionControlEdit class object in the main validation loop.
"""

from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
import copy
import math
from einops import rearrange
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from i2vedit.prompt_attention.visualization import (
    show_cross_attention, 
    show_self_attention_comp, 
    show_self_attention, 
    show_self_attention_distance, 
    calculate_attention_mask, 
    show_avg_difference_maps
)
from i2vedit.prompt_attention.attention_store import AttentionStore, AttentionControl
from i2vedit.prompt_attention.attention_register import register_attention_control

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControlEdit(AttentionStore, abc.ABC):
    """Decide self or cross-attention. Call the reweighting cross attention module

    Args:
        AttentionStore (_type_): ([1, 4, 8, 64, 64])
        abc (_type_): [8, 8, 1024, 77]
    """
    def get_all_last_latents(self, overlay_size):
        return [latents[:,-overlay_size:,...] for latents in self.latents_store]
        
    
    def step_callback(self, x_t):
        x_t = super().step_callback(x_t)
        x_t_device = x_t.device
        x_t_dtype = x_t.dtype

#        if self.previous_latents is not None:
#            # replace latents
#            step_in_store = self.cur_step - 1
#            previous_latents = self.previous_latents[step_in_store]
#            x_t[:,:len(previous_latents),...] = previous_latents.to(x_t_device, x_t_dtype)
        if self.latent_blend:

            avg_attention = self.get_average_attention()
            masks = []
            for key in avg_attention:
                if 'down' in key and 'mask' in key:
                    for attn in avg_attention[key]:
                        if attn.shape[-2] == 8 * 9:
                            masks.append( attn )
            mask = sum(masks) / len(masks)
            mask[mask > 0.2] = 1.0
            if self.use_inversion_attention and self.additional_attention_store is not None:
                step_in_store = len(self.additional_attention_store.latents_store) - self.cur_step
            elif self.additional_attention_store is None:
                pass
            else:
                step_in_store = self.cur_step - 1
                
            inverted_latents = self.additional_attention_store.latents_store[step_in_store]
            inverted_latents = inverted_latents.to(device =x_t_device, dtype=x_t_dtype)

            x_t = (1 - mask) * inverted_latents + mask * x_t

        self.step_in_store_atten_dict = None

        return x_t
        
    def replace_self_attention(self, attn_base, attn_replace, reshaped_mask=None, key=None):
        
        target_device = attn_replace.device
        target_dtype  = attn_replace.dtype
        attn_base = attn_base.to(target_device, dtype=target_dtype)

        if "temporal" in key:

            if self.control_mode["temporal_self"] == "copy_v2":
                if self.cur_step < int(self.temporal_step_thr[0] * self.num_steps):
                    return attn_base
                if self.cur_step >= int(self.temporal_step_thr[1] * self.num_steps):
                    return attn_replace
                if ('down' in key and self.current_pos<4) or \
                   ('up' in key and self.current_pos>1):
                    return attn_replace
                return attn_base

            else:
                raise NotImplementedError

        elif "spatial" in key:
            
            raise NotImplementedError
    
    def replace_cross_attention(self, attn_base, attn_replace, key=None):
        raise NotImplementedError
    
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)

        if 'mask' in place_in_unet:
            return attn

        if (not is_cross and 'temporal' in place_in_unet and (self.cur_step < self.num_temporal_self_replace[0] or self.cur_step >=self.num_temporal_self_replace[1])):
            if self.control_mode["temporal_self"] == "copy" or \
               self.control_mode["temporal_self"] == "copy_v2":
                return attn

        if (not is_cross and 'spatial' in place_in_unet and (self.cur_step < self.num_spatial_self_replace[0] or self.cur_step >=self.num_spatial_self_replace[1])):
            if self.control_mode["spatial_self"] == "copy":
                return attn

        if (is_cross and (self.cur_step < self.num_cross_replace[0] or self.cur_step >= self.num_cross_replace[1])):
            return attn
        
        if True:#'temporal' in place_in_unet:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            current_pos = self.attention_position_counter_dict[key]

            if self.use_inversion_attention and self.additional_attention_store is not None:
                step_in_store = len(self.additional_attention_store.attention_store_all_step) - self.cur_step -1
            elif self.additional_attention_store is None:
                return attn

            else:
                step_in_store = self.cur_step
                
            step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
            
            if isinstance(step_in_store_atten_dict, str):
                if self.step_in_store_atten_dict is None: 
                    step_in_store_atten_dict = torch.load(step_in_store_atten_dict)
                    self.step_in_store_atten_dict = step_in_store_atten_dict
                else:
                    step_in_store_atten_dict = self.step_in_store_atten_dict
            
            # Note that attn is append to step_store, 
            # if attn is get through clean -> noisy, we should inverse it
            #print(key)
            attn_base = step_in_store_atten_dict[key][current_pos]          
            self.current_pos = current_pos
            
            self.update_attention_position_dict(key)
            # save in format of [temporal, head, resolution, text_embedding]
            attn_base, attn_replace = attn_base, attn

            if not is_cross:
                attn = self.replace_self_attention(attn_base, attn_replace, None, key)

            #elif is_cross and (self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]):
            elif is_cross:
                attn = self.replace_cross_attention(attn_base, attn_replace, key)

            return attn

        else:
            
            raise NotImplementedError("Due to CUDA RAM limit, direct replace functions for spatial  are not implemented.")

    def between_steps(self):

        super().between_steps()



        self.step_store = self.get_empty_store()
        
        self.attention_position_counter_dict = {
            'down_spatial_q_cross': 0,
            'mid_spatial_q_cross': 0,
            'up_spatial_q_cross': 0,
            'down_spatial_k_cross': 0,
            'mid_spatial_k_cross': 0,
            'up_spatial_k_cross': 0,
            'down_spatial_mask_cross': 0,
            'mid_spatial_mask_cross': 0,
            'up_spatial_mask_cross': 0,
            'down_spatial_q_self': 0,
            'mid_spatial_q_self': 0,
            'up_spatial_q_self': 0,
            'down_spatial_k_self': 0,
            'mid_spatial_k_self': 0,
            'up_spatial_k_self': 0,
            'down_spatial_mask_self': 0,
            'mid_spatial_mask_self': 0,
            'up_spatial_mask_self': 0,
            'down_temporal_cross': 0,
            'mid_temporal_cross': 0,
            'up_temporal_cross': 0,
            'down_temporal_self': 0,
            'mid_temporal_self': 0,
            'up_temporal_self': 0
        }        
        return 

    def __init__(self, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 temporal_self_replace_steps: Union[float, Tuple[float, float]],
                 spatial_self_replace_steps: Union[float, Tuple[float, float]],
                 control_mode={"temporal_self":"copy","spatial_self":"copy"},
                 spatial_attention_chunk_size = 1,
                 additional_attention_store: AttentionStore =None,
                 use_inversion_attention: bool=False,
                 save_self_attention: bool=True,
                 save_latents: bool=True,
                 disk_store=False,
                 *args, **kwargs
                 ):
        super(AttentionControlEdit, self).__init__(
            save_self_attention=save_self_attention,
            save_latents=save_latents,
            disk_store=disk_store)
        self.additional_attention_store = additional_attention_store
        if type(temporal_self_replace_steps) is float:
            temporal_self_replace_steps = 0, temporal_self_replace_steps
        if type(spatial_self_replace_steps) is float:
            spatial_self_replace_steps = 0, spatial_self_replace_steps
        if type(cross_replace_steps) is float:
            cross_replace_steps = 0, cross_replace_steps
        self.num_temporal_self_replace = int(num_steps * temporal_self_replace_steps[0]), int(num_steps * temporal_self_replace_steps[1])
        self.num_spatial_self_replace = int(num_steps * spatial_self_replace_steps[0]), int(num_steps * spatial_self_replace_steps[1])
        self.num_cross_replace = int(num_steps * cross_replace_steps[0]), int(num_steps * cross_replace_steps[1])
        self.control_mode = control_mode
        self.spatial_attention_chunk_size = spatial_attention_chunk_size
        self.step_in_store_atten_dict = None
        # We need to know the current position in attention
        self.prev_attention_key_name = 0
        self.use_inversion_attention = use_inversion_attention
        self.attention_position_counter_dict = {
            'down_spatial_q_cross': 0,
            'mid_spatial_q_cross': 0,
            'up_spatial_q_cross': 0,
            'down_spatial_k_cross': 0,
            'mid_spatial_k_cross': 0,
            'up_spatial_k_cross': 0,
            'down_spatial_mask_cross': 0,
            'mid_spatial_mask_cross': 0,
            'up_spatial_mask_cross': 0,
            'down_spatial_q_self': 0,
            'mid_spatial_q_self': 0,
            'up_spatial_q_self': 0,
            'down_spatial_k_self': 0,
            'mid_spatial_k_self': 0,
            'up_spatial_k_self': 0,
            'down_spatial_mask_self': 0,
            'mid_spatial_mask_self': 0,
            'up_spatial_mask_self': 0,
            'down_temporal_cross': 0,
            'mid_temporal_cross': 0,
            'up_temporal_cross': 0,
            'down_temporal_self': 0,
            'mid_temporal_self': 0,
            'up_temporal_self': 0
        }
        self.mask_thr = kwargs.get("mask_thr", 0.35)
        self.latent_blend = kwargs.get('latent_blend', False)

        self.temporal_step_thr = kwargs.get("temporal_step_thr", [0.4,0.8])
        self.num_steps = num_steps

    def spatial_attention_control(
            self, place_in_unet, attention_type, is_cross, 
            q, k, v, attn_mask, dropout_p=0.0, is_causal=False
        ):

        return self.spatial_attention_matching(
            place_in_unet, attention_type, is_cross,
            q, k, v, attn_mask, dropout_p=0.0, is_causal=False,
            mode = self.control_mode["spatial_self"]
        )


    def spatial_attention_matching(
            self, place_in_unet, attention_type, is_cross, 
            q, k, v, attn_mask, dropout_p=0.0, is_causal=False,
            mode = "matching"
        ):
        place_in_unet = f"{place_in_unet}_{attention_type}"
        with sdp_kernel(**BACKEND_MAP[None]):
#            print("register", q.shape, k.shape, v.shape)

            # fetch inversion q and k
            key_q = f"{place_in_unet}_q_{'cross' if is_cross else 'self'}"
            key_k = f"{place_in_unet}_k_{'cross' if is_cross else 'self'}"
            current_pos_q = self.attention_position_counter_dict[key_q]
            current_pos_k = self.attention_position_counter_dict[key_k]

            if self.use_inversion_attention and self.additional_attention_store is not None:
                step_in_store = len(self.additional_attention_store.attention_store_all_step) - self.cur_step -1
            else:
                step_in_store = self.cur_step
                
            step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
            
            if isinstance(step_in_store_atten_dict, str):
                if self.step_in_store_atten_dict is None: 
                    step_in_store_atten_dict = torch.load(step_in_store_atten_dict)
                    self.step_in_store_atten_dict = step_in_store_atten_dict
                else:
                    step_in_store_atten_dict = self.step_in_store_atten_dict
            
            q0s = step_in_store_atten_dict[key_q][current_pos_q].to(q.device)          
            k0s = step_in_store_atten_dict[key_k][current_pos_k].to(k.device)
            
            self.update_attention_position_dict(key_q)
            self.update_attention_position_dict(key_k)

            qs, ks, vs = q, k, v
            
            h = q.shape[1]
            res = int(np.sqrt(q.shape[-2] / (9*16))) 
            #res = int(np.sqrt(q.shape[-2] / (8*14))) 
            bs = self.spatial_attention_chunk_size
            if bs is None: bs = qs.shape[0]
            N = qs.shape[0] // bs
            assert qs.shape[0] % bs == 0
            i1st, n1st = qs.shape[0]//2//bs, qs.shape[0]//2%bs
            outs = []
            masks = []

            # this might reduce time costs but will introduce inaccurate motions
#            if current_pos_q >= 6 and 'up' in key_q: 
#                return F.scaled_dot_product_attention(
#                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
#                )

            for i in range(N):
                q = qs[i*bs:(i+1)*bs,...].type(torch.float32)
                k = ks[i*bs:(i+1)*bs,...].type(torch.float32)
                v = vs[i*bs:(i+1)*bs,...].type(torch.float32)

                q, k, v = map(lambda t: rearrange(t, "b h n d -> (b h) n d"), (q, k, v))

                with torch.autocast("cuda", enabled=False):
                    attention_scores = torch.baddbmm(
                        torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
                        q,
                        k.transpose(-1, -2),
                        beta=0,
                        alpha=1 / math.sqrt(q.size(-1)),
                    )

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    attention_scores = attention_scores + attn_mask

                attention_probs = attention_scores.softmax(dim=-1).to(vs.dtype)

                # only compute conditional output
                if i >= N//2:

                    q0 = q0s[(i-N//2)*bs:(i-N//2+1)*bs,...].type(torch.float32)
                    k0 = k0s[(i-N//2)*bs:(i-N//2+1)*bs,...].type(torch.float32)
                    
                    q0, k0 = map(lambda t: rearrange(t, "b h n d -> (b h) n d"), (q0, k0))

                    with torch.autocast("cuda", enabled=False):
                        attention_scores_0 = torch.baddbmm(
                            torch.empty(q0.shape[0], q0.shape[1], k0.shape[1], dtype=q0.dtype, device=q0.device),
                            q0,
                            k0.transpose(-1, -2),
                            beta=0,
                            alpha=1 / math.sqrt(q0.size(-1)),
                        )
                    
                    attention_probs_0 = attention_scores_0.softmax(dim=-1).to(vs.dtype)

                    attention_probs, attention_probs_0 = \
                        map(lambda t: rearrange(t, "(b h) n d -> b h n d", h=h), 
                            (attention_probs, attention_probs_0))

                    if mode == "masked_copy":
                                
                        mask = torch.sum(
                            torch.mean(
                                torch.abs(attention_probs_0 - attention_probs), 
                                dim=1
                            ), 
                            dim=2
                        ).reshape(bs,1,-1,1).clamp(0,2)/2.0
                        mask_thr = (self.mask_thr[1]-self.mask_thr[0]) / (qs.shape[0]//2)*(i-N//2) + self.mask_thr[0]
                        mask_tmp = mask.clone()
                        mask[mask>=mask_thr] = 1.0
                        masks.append(mask)

                        # apply mask
                        attention_probs = (1 - mask) * attention_probs_0 + mask * attention_probs

                    else:
                        raise NotImplementedError

                    attention_probs = rearrange(attention_probs, "b h n d -> (b h) n d")

                # compute attention output
                hidden_states = torch.bmm(attention_probs, v)

                # reshape hidden_states
                hidden_states = rearrange(hidden_states, "(b h) n d -> b h n d", h=h)

                outs.append(hidden_states)

            if mode == "masked_copy":

                #masks = rearrange(torch.cat(masks, 0), "b 1 (h w) 1 -> h (b w)", h=res*9)
                masks = torch.cat(masks, 0)
                #print(f"{place_in_unet}_masked_copy")
                # save mask
                _ = self.__call__(masks, is_cross, f"{place_in_unet}_mask")

            return torch.cat(outs, 0)

class ConsistencyAttentionControl(AttentionStore, abc.ABC):
    """Decide self or cross-attention. Call the reweighting cross attention module

    Args:
        AttentionStore (_type_): ([1, 4, 8, 64, 64])
        abc (_type_): [8, 8, 1024, 77]
    """
    def step_callback(self, x_t):
        x_t = super().step_callback(x_t)
        x_t_device = x_t.device
        x_t_dtype = x_t.dtype

#        if self.previous_latents is not None:
#            # replace latents
#            step_in_store = self.cur_step - 1
#            previous_latents = self.previous_latents[step_in_store]
#            x_t[:,:len(previous_latents),...] = previous_latents.to(x_t_device, x_t_dtype)

        self.step_in_store_atten_dict = None

        return x_t
        
    def update_attention_position_dict(self, current_attention_key):
        self.attention_position_counter_dict[current_attention_key] +=1

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet)

        self.cur_att_layer += 1

        return attn

    def set_cur_step(self, step: int = 0):
        self.cur_step = step

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(ConsistencyAttentionControl, self).forward(attn, is_cross, place_in_unet)

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        current_pos = self.attention_position_counter_dict[key]

        if self.use_inversion_attention and self.additional_attention_store is not None:
            step_in_store = len(self.additional_attention_store.attention_store_all_step) - self.cur_step -1
        elif self.additional_attention_store is None:
            return attn

        else:
            step_in_store = self.cur_step
            
        step_in_store_atten_dict = self.additional_attention_store.attention_store_all_step[step_in_store]
        
        if isinstance(step_in_store_atten_dict, str):
            if self.step_in_store_atten_dict is None: 
                step_in_store_atten_dict = torch.load(step_in_store_atten_dict)
                self.step_in_store_atten_dict = step_in_store_atten_dict
            else:
                step_in_store_atten_dict = self.step_in_store_atten_dict
        
        # Note that attn is append to step_store, 
        # if attn is get through clean -> noisy, we should inverse it
        #print("consistency", key)
        attn_base = step_in_store_atten_dict[key][current_pos].to(attn.device, attn.dtype)          
        attn_base = attn_base.detach()
        
        self.update_attention_position_dict(key)
        # save in format of [temporal, head, resolution, text_embedding]
        
        attn = torch.cat([attn_base, attn], dim=2)

        return attn

    @staticmethod
    def get_empty_store():
        return {
                "down_temporal_k_self": [],  "mid_temporal_k_self": [],  "up_temporal_k_self": [],
                "down_temporal_v_self": [],  "mid_temporal_v_self": [],  "up_temporal_v_self": []
               }

    def between_steps(self):

        super().between_steps()

        self.step_store = self.get_empty_store()
        
        self.attention_position_counter_dict = {
            'down_temporal_k_self': 0,
            'mid_temporal_k_self': 0,
            'up_temporal_k_self': 0,
            'down_temporal_v_self': 0,
            'mid_temporal_v_self': 0,
            'up_temporal_v_self': 0
        }        
        return 

    def __init__(self, 
                 additional_attention_store: AttentionStore =None,
                 use_inversion_attention: bool=False,
                 load_attention_store: str = None,
                 save_self_attention: bool=True,
                 save_latents: bool=True,
                 disk_store=False,
                 store_path:str=None
                 ):
        super(ConsistencyAttentionControl, self).__init__(
            save_self_attention=save_self_attention,
            load_attention_store=load_attention_store,
            save_latents=save_latents,
            disk_store=disk_store,
            store_path=store_path
        )

        self.additional_attention_store = additional_attention_store
        self.step_in_store_atten_dict = None
        # We need to know the current position in attention
        self.use_inversion_attention = use_inversion_attention
        self.attention_position_counter_dict = {
            'down_temporal_k_self': 0,
            'mid_temporal_k_self': 0,
            'up_temporal_k_self': 0,
            'down_temporal_v_self': 0,
            'mid_temporal_v_self': 0,
            'up_temporal_v_self': 0
        }
            
            

def make_controller(
                    cross_replace_steps: Dict[str, float], self_replace_steps: float=0.0, 
                    additional_attention_store=None, use_inversion_attention = False,
                    NUM_DDIM_STEPS=None,
                    save_path = None,
                    save_self_attention = True,
                    disk_store = False
                    ) -> AttentionControlEdit:
    controller = AttentionControlEdit(NUM_DDIM_STEPS, 
                                  cross_replace_steps=cross_replace_steps, 
                                  self_replace_steps=self_replace_steps, 
                                  additional_attention_store=additional_attention_store,
                                  use_inversion_attention = use_inversion_attention,
                                  save_self_attention = save_self_attention,
                                  disk_store=disk_store
                                  )
    return controller

