import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import imageio
import numpy as np
from PIL import Image
from scipy.stats import anderson
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import CLIPEncoder

import diffusers
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.unet_3d_blocks import \
            (CrossAttnDownBlockSpatioTemporal, 
             DownBlockSpatioTemporal,
             CrossAttnUpBlockSpatioTemporal,
             UpBlockSpatioTemporal)
from i2vedit.utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset, \
    pad_with_ratio, return_to_original_res
from einops import rearrange, repeat
from i2vedit.utils.lora_handler import LoraHandler
from i2vedit.utils.lora import extract_lora_child_module
from i2vedit.utils.euler_utils import euler_inversion
from i2vedit.utils.svd_util import SmoothAreaRandomDetection

from i2vedit.data import VideoIO, SingleClipDataset, ResolutionControl
#from utils.model_utils import load_primary_models
from i2vedit.utils.euler_utils import inverse_video
from i2vedit.train import train_motion_lora, load_images_from_list
from i2vedit.inference import initialize_pipeline
from i2vedit.utils.model_utils import P2PEulerDiscreteScheduler, P2PStableVideoDiffusionPipeline
from i2vedit.prompt_attention import attention_util

def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"i2vedit_{now}")
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))
    return out_dir



def main(
    pretrained_model_path: str,
    data_params: Dict,
    train_motion_lora_params: Dict,
    sarp_params: Dict,
    attention_matching_params: Dict,
    long_video_params: Dict = {"mode": "skip-interval"},
    use_sarp: bool = True,
    use_motion_lora: bool = True,
    train_motion_lora_only: bool = False,
    retrain_motion_lora: bool = True,
    use_inversed_latents: bool = True,
    use_attention_matching: bool = True,
    use_consistency_attention_control: bool = False,
    output_dir: str = "./outputs",
    num_steps: int = 25,
    device: str = "cuda",
    seed: int = 23,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    dtype: str = 'fp16',
    load_from_last_frames_latents: List[str] = None,
    save_last_frames: bool = True,
    visualize_attention_store: bool = False,
    visualize_attention_store_steps: List[int] = None,
    use_latent_blend: bool = False,
    use_previous_latent_for_train: bool = False,
    use_latent_noise: bool = True,
    load_from_previous_consistency_store_controller: str = None,
    load_from_previous_consistency_edit_controller: List[str] = None
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32

    # create folder
    output_dir = create_output_folders(output_dir, config)

    # prepare video data 
    data_params["output_dir"] = output_dir
    data_params["device"] = device

    videoio = VideoIO(**data_params, dtype=dtype)

    # smooth area random perturbation
    if use_sarp:
        sard = SmoothAreaRandomDetection(device, dtype=torch.float32)
    else:
        sard = None
    
    keyframe = None
    previous_last_frames = load_images_from_list(data_params.keyframe_paths)
    consistency_train_controller = None 

    if load_from_last_frames_latents is not None:
        previous_last_frames_latents = [torch.load(thpath).to(device) for thpath in load_from_last_frames_latents]
    else:
        previous_last_frames_latents = [None,] * len(previous_last_frames)

    if use_consistency_attention_control and load_from_previous_consistency_store_controller is not None:
        previous_consistency_store_controller = attention_util.ConsistencyAttentionControl(
             additional_attention_store=None,
             use_inversion_attention=False,
             save_self_attention=True,
             save_latents=False,
             disk_store=True,
             load_attention_store=os.path.join(load_from_previous_consistency_store_controller, "clip_0")
        )
    else:
        previous_consistency_store_controller = None

    previous_consistency_edit_controller_list = [None,] * len(previous_last_frames)
    if use_consistency_attention_control and load_from_previous_consistency_edit_controller is not None:
        for i in range(len(load_from_previous_consistency_edit_controller)):
            previous_consistency_edit_controller_list[i] = attention_util.ConsistencyAttentionControl(
                 additional_attention_store=None,
                 use_inversion_attention=False,
                 save_self_attention=True,
                 save_latents=False,
                 disk_store=True,
                 load_attention_store=os.path.join(load_from_previous_consistency_edit_controller[i], "clip_0")
            )


    # read data and process
    for clip_id, video in enumerate(videoio.read_video_iter()):
        if clip_id >= data_params.get("end_clip_id", 9):
            break
        if clip_id < data_params.get("begin_clip_id", 0):
            continue
        video = video.unsqueeze(0)
        
        resctrl = ResolutionControl(video.shape[-2:], data_params.output_res, data_params.pad_to_fit, fill=-1)

        # update keyframe and edited keyframe
        if long_video_params.mode == "skip-interval":
            assert data_params.overlay_size > 0 
            # save the first frame as the keyframe for cross-attention
            #if clip_id == 0:
            firstframe = video[:,0:1,:,:,:]
            keyframe = video[:,0:1,:,:,:]
            edited_keyframes = copy.deepcopy(previous_last_frames)
            edited_firstframes = edited_keyframes 
            #edited_firstframes = load_images_from_list(data_params.keyframe_paths)

        elif long_video_params.mode == "auto-regressive":
            assert data_params.overlay_size == 1 
            firstframe = video[:,0:1,:,:,:]
            keyframe = video[:,0:1,:,:,:]
            edited_keyframes = copy.deepcopy(previous_last_frames)
            edited_firstframes = edited_keyframes

        # register for unet, perform inversion
        load_attention_store = None
        if use_attention_matching:
            assert use_inversed_latents, "inversion is disabled."
            if attention_matching_params.get("load_attention_store") is not None:
                load_attention_store = os.path.join(attention_matching_params.get("load_attention_store"), f"clip_{clip_id}")
                if not os.path.exists(load_attention_store):
                    print(f"Load {load_attention_store} failed, folder doesn't exists.")
                    load_attention_store = None

            store_controller = attention_util.AttentionStore(
                disk_store=attention_matching_params.disk_store,
                save_latents = use_latent_blend,
                save_self_attention=True,
                load_attention_store=load_attention_store,
                store_path=os.path.join(output_dir, "attention_store", f"clip_{clip_id}")
            )
            print("store_controller.store_dir:", store_controller.store_dir)
        else:
            store_controller = None

        load_consistency_attention_store = None
        if use_consistency_attention_control:
            if clip_id==0 and attention_matching_params.get("load_consistency_attention_store") is not None:
                load_consistency_attention_store = os.path.join(attention_matching_params.get("load_consistency_attention_store"), f"clip_{clip_id}")
                if not os.path.exists(load_consistency_attention_store):
                     print(f"Load {load_consistency_attention_store} failed, folder doesn't exists.")
                     load_consistency_attention_store = None

            consistency_store_controller = attention_util.ConsistencyAttentionControl(
                 additional_attention_store=previous_consistency_store_controller,
                 use_inversion_attention=False,
                 save_self_attention=(clip_id==0),
                 load_attention_store=load_consistency_attention_store,
                 save_latents=False,
                 disk_store=True,
                 store_path=os.path.join(output_dir, "consistency_attention_store", f"clip_{clip_id}")
            )
            print("consistency_store_controller.store_dir:", consistency_store_controller.store_dir)
        else:
            consistency_store_controller = None

        if train_motion_lora_only:
            assert use_motion_lora and retrain_motion_lora, "use_motion_lora/retrain_motion_lora should be enbled to train motion lora only."

        # perform smooth area random perturbation     
        if use_inversed_latents:
            print("begin inversion sampling for inference...")
            inversion_noise = inverse_video(
                pretrained_model_path, 
                video, 
                keyframe,
                firstframe,
                num_steps, 
                resctrl, 
                sard, 
                enable_xformers_memory_efficient_attention,
                enable_torch_2_attn,
                store_controller = store_controller,
                consistency_store_controller = consistency_store_controller,
                find_modules=attention_matching_params.registered_modules if load_attention_store is None else {},
                consistency_find_modules=long_video_params.registered_modules if load_consistency_attention_store is None else {},
               # dtype=dtype,
                **sarp_params,
            )
        else:
            if use_motion_lora and retrain_motion_lora:
                assert not any([np > 0 for np in train_motion_lora_params.validation_data.noise_prior]), "inversion noise is not calculated but validation during motion lora training aims to use inversion noise as input latents."
            inversion_noise = None


        if use_motion_lora:
            if retrain_motion_lora:
                if use_consistency_attention_control:
                    if data_params.output_res[0] != train_motion_lora_params.train_data.height or \
                       data_params.output_res[1] != train_motion_lora_params.train_data.width:
                        if consistency_train_controller is None:
                            load_consistency_train_attention_store = None
                            if attention_matching_params.get("load_consistency_train_attention_store") is not None:
                                load_consistency_train_attention_store = os.path.join(attention_matching_params.get("load_consistency_train_attention_store"), f"clip_0")
                                if not os.path.exists(load_consistency_train_attention_store):
                                    print(f"Load {load_consistency_train_attention_store} failed, folder doesn't exists.")
                                    load_consistency_train_attention_store = None
                            if load_consistency_train_attention_store is None and clip_id > 0:
                                raise IOError(f"load_consistency_train_attention_store can't be None for clip {clip_id}.")
                            consistency_train_controller = attention_util.ConsistencyAttentionControl(
                                 additional_attention_store=None,
                                 use_inversion_attention=False,
                                 save_self_attention=True,
                                 load_attention_store=load_consistency_train_attention_store,
                                 save_latents=False,
                                 disk_store=True,
                                 store_path=os.path.join(output_dir, "consistency_train_attention_store", "clip_0")
                            )
                            print("consistency_train_controller.store_dir:", consistency_train_controller.store_dir)
                            resctrl_train = ResolutionControl(
                                video.shape[-2:], 
                                (train_motion_lora_params.train_data.height,train_motion_lora_params.train_data.width), 
                                data_params.pad_to_fit, fill=-1
                            )
                            print("begin inversion sampling for training...")
                            inversion_noise_train = inverse_video(
                                pretrained_model_path, 
                                video, 
                                keyframe,
                                firstframe,
                                num_steps, 
                                resctrl_train, 
                                sard, 
                                enable_xformers_memory_efficient_attention,
                                enable_torch_2_attn,
                                store_controller = None,
                                consistency_store_controller = consistency_train_controller,
                                find_modules={},
                                consistency_find_modules=long_video_params.registered_modules if long_video_params.get("load_attention_store") is None else {},
                               # dtype=dtype,
                                **sarp_params,
                            )
                    else:
                        if consistency_train_controller is None:
                            consistency_train_controller = consistency_store_controller
                else:
                    consistency_train_controller = None

            if retrain_motion_lora:
                train_dataset = SingleClipDataset(
                    inversion_noise=inversion_noise,
                    video_clip=video,
                    keyframe=((ToTensor()(previous_last_frames[0])-0.5)/0.5).unsqueeze(0).unsqueeze(0) if use_previous_latent_for_train else keyframe,
                    keyframe_latent=previous_last_frames_latents[0] if use_previous_latent_for_train else None,
                    firstframe=firstframe,
                    height=train_motion_lora_params.train_data.height,
                    width=train_motion_lora_params.train_data.width,
                    use_data_aug=train_motion_lora_params.train_data.get("use_data_aug"),
                    pad_to_fit=train_motion_lora_params.train_data.get("pad_to_fit", False)
                )
                train_motion_lora_params.validation_data.num_inference_steps = num_steps
                train_motion_lora(
                    pretrained_model_path,
                    output_dir, 
                    train_dataset, 
                    edited_firstframes=edited_firstframes,
                    validation_images=edited_keyframes,
                    validation_images_latents=previous_last_frames_latents,
                    seed=seed,
                    clip_id=clip_id,
                    consistency_edit_controller_list=previous_consistency_edit_controller_list,
                    consistency_controller=consistency_train_controller if clip_id!=0 else None,
                    consistency_find_modules=long_video_params.registered_modules,
                    enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
                    enable_torch_2_attn=enable_torch_2_attn,
                    **train_motion_lora_params
                )

            if train_motion_lora_only:
                if not use_consistency_attention_control:
                    continue

            # choose and load motion lora
            best_checkpoint_index = attention_matching_params.get("best_checkpoint_index", 250)
            if retrain_motion_lora:
                lora_dir = f"{os.path.join(output_dir,'train_motion_lora')}/clip_{clip_id}"
            else:
                lora_dir = os.path.join(attention_matching_params.lora_dir, f"clip_{clip_id}") 
            lora_path = f"{lora_dir}/checkpoint-{best_checkpoint_index}/temporal/lora"
            assert os.path.exists(lora_path), f"lora path: {lora_path} doesn't exist!"
                
            lora_rank = train_motion_lora_params.lora_rank
            lora_scale = attention_matching_params.get("lora_scale", 1.0)
                
            # prepare models
            pipe = initialize_pipeline(
                pretrained_model_path, 
                device, 
                enable_xformers_memory_efficient_attention, 
                enable_torch_2_attn,
                lora_path, 
                lora_rank, 
                lora_scale,
                load_spatial_lora = False #(clip_id != 0)
            ).to(device, dtype=dtype)
        else:
            pipe = P2PStableVideoDiffusionPipeline.from_pretrained(
                pretrained_model_path
            ).to(device, dtype=dtype)

        if use_attention_matching or use_consistency_attention_control:
            pipe.scheduler = P2PEulerDiscreteScheduler.from_config(pipe.scheduler.config)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        previous_last_frames = []

        editing_params = [item for name, item in attention_matching_params.params.items()]
        with torch.no_grad():
            with torch.autocast(device, dtype=dtype):
                for kf_id, (edited_keyframe, editing_param) in enumerate(zip(edited_keyframes, editing_params)):
                    print(kf_id, editing_param)

                    # control resolution
                    iw, ih = edited_keyframe.size
                    resctrl = ResolutionControl(
                        (ih, iw), 
                        data_params.output_res, 
                        data_params.pad_to_fit, 
                        fill=0
                    )
                    edited_keyframe = resctrl(edited_keyframe)
                    edited_firstframe = resctrl(edited_firstframes[kf_id])

                    # control attention
                    pipe.scheduler.controller = []
                    if use_attention_matching:
                        edit_controller = attention_util.AttentionControlEdit(
                            num_steps = num_steps, 
                            cross_replace_steps = attention_matching_params.cross_replace_steps,
                            temporal_self_replace_steps = attention_matching_params.temporal_self_replace_steps,
                            spatial_self_replace_steps = attention_matching_params.spatial_self_replace_steps,
                            mask_thr = editing_param.get("mask_thr", 0.35),
                            temporal_step_thr = editing_param.get("temporal_step_thr", [0.5,0.8]),
                            control_mode = attention_matching_params.control_mode,
                            spatial_attention_chunk_size = attention_matching_params.get("spatial_attention_chunk_size", 1),
                            additional_attention_store = store_controller,
                            use_inversion_attention = True,
                            save_self_attention = False,
                            save_latents = False,
                            latent_blend = use_latent_blend,
                            disk_store = attention_matching_params.disk_store
                        )
                        pipe.scheduler.controller.append(edit_controller) 
                    else:
                        edit_controller = None

                    if use_consistency_attention_control:
                        consistency_edit_controller = attention_util.ConsistencyAttentionControl(
                             additional_attention_store=previous_consistency_edit_controller_list[kf_id],
                             use_inversion_attention=False,
                             save_self_attention=(clip_id==0),
                             save_latents=False,
                             disk_store=True,
                             store_path=os.path.join(output_dir, f"consistency_edit{kf_id}_attention_store", f"clip_{clip_id}")
                        )
                        pipe.scheduler.controller.append(consistency_edit_controller) 
                    else:
                        consistency_edit_controller = None

                    if use_attention_matching or use_consistency_attention_control:
                        attention_util.register_attention_control(
                            pipe.unet,
                            edit_controller,
                            consistency_edit_controller,
                            find_modules=attention_matching_params.registered_modules,
                            consistency_find_modules=long_video_params.registered_modules
                        )

                    # should be reorganized to perform attention control
                    edited_output = pipe(
                        edited_keyframe,
                        edited_firstframe=edited_firstframe,
                        image_latents=previous_last_frames_latents[kf_id],
                        width=data_params.output_res[1],
                        height=data_params.output_res[0],
                        num_frames=video.shape[1],
                        num_inference_steps=num_steps,
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=data_params.output_fps,
                        noise_aug_strength=0.02,
                        max_guidance_scale=attention_matching_params.get("max_guidance_scale", 2.5),
                        generator=generator,
                        latents=inversion_noise
                    )
                    edited_video = [img for sublist in edited_output.frames for img in sublist]
                    edited_video_latents = edited_output.latents
                    
                    # callback to replace frames
                    videoio.write_video(edited_video, kf_id, resctrl)

                    # save previous frames
                    if long_video_params.mode == "skip-interval":
                        #previous_latents[kf_id] = edit_controller.get_all_last_latents(data_params.overlay_size)
                        previous_last_frames.append( resctrl.callback(edited_video[-1]) )
                        if use_latent_noise:
                            previous_last_frames_latents[kf_id] = edited_video_latents[:,-1:,:,:,:]
                        else:
                            previous_last_frames_latents[kf_id] = None
                    elif long_video_params.mode == "auto-regressive":
                        previous_last_frames.append( resctrl.callback(edited_video[-1]) )
                        if use_latent_noise:
                            previous_last_frames_latents[kf_id] = edited_video_latents[:,-1:,:,:,:]
                        else:
                            previous_last_frames_latents[kf_id] = None

                    # save last frames for convenient
                    if save_last_frames:
                        try:
                            fname = os.path.join(output_dir, f"clip_{clip_id}_lastframe_{kf_id}")
                            previous_last_frames[kf_id].save(fname+".png")
                            if use_latent_noise:
                                torch.save(previous_last_frames_latents[kf_id], fname+".pt")
                        except:
                            print("save fail")

                    if use_attention_matching or use_consistency_attention_control:
                        attention_util.register_attention_control(
                            pipe.unet,
                            edit_controller,
                            consistency_edit_controller,
                            find_modules=attention_matching_params.registered_modules,
                            consistency_find_modules=long_video_params.registered_modules,
                            undo=True
                        )
                        if edit_controller is not None:
                            if visualize_attention_store:
                                vis_save_path = os.path.join(output_dir, "visualization", f"{kf_id}", f"clip_{clip_id}")
                                os.makedirs(vis_save_path, exist_ok=True)
                                attention_util.show_avg_difference_maps(
                                    edit_controller,
                                    save_path = vis_save_path
                                )
                                assert visualize_attention_store_steps is not None
                                attention_util.show_self_attention(
                                    edit_controller,
                                    steps = visualize_attention_store_steps, 
                                    save_path = vis_save_path,
                                    inversed = False
                                )
                            edit_controller.delete()
                            del edit_controller
                    if use_consistency_attention_control:
                        if clip_id == 0:
                            previous_consistency_edit_controller_list[kf_id] = consistency_edit_controller
                        else:
                            consistency_edit_controller.delete()
                            del consistency_edit_controller
                        print(f"previous_consistency_edit_controller_list[{kf_id}]", previous_consistency_edit_controller_list[kf_id].store_dir)


        if use_attention_matching:
            del store_controller

        if use_consistency_attention_control and clip_id == 0:
            previous_consistency_store_controller = consistency_store_controller
        
    videoio.close()

    if use_consistency_attention_control:
        print("consistency_store_controller for clip 0:", previous_consistency_store_controller.store_dir) 
        if retrain_motion_lora:
            print("consistency_train_controller for clip 0:", consistency_train_controller.store_dir)
        for kf_id in range(len(previous_consistency_edit_controller_list)):
            print(f"previous_consistency_edit_controller_list[{kf_id}]:", previous_consistency_edit_controller_list[kf_id].store_dir)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/svdedit/item2_2.yaml')
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))












































