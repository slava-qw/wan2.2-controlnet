import os
import glob
import random

import torch
import numpy as np
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import load_video
from torchvision import transforms


class ControlnetDataset(Dataset):
    def __init__(
            self, 
            latents_dir,
            text_embeds_dir,
            controlnet_video_dir,
            control_video_dir,
            downscale_coef=16,
            stage_2_training=False,
            sample_type='non_uniform',
            seed=None,
        ):
        self.latents_dir = latents_dir
        self.text_embeds_dir = text_embeds_dir
        self.controlnet_video_dir = controlnet_video_dir
        self.control_video_dir = control_video_dir

        videos_paths = glob.glob(os.path.join(self.controlnet_video_dir, '*.mp4'))
        self.videos_names = [os.path.basename(x) for x in videos_paths]
        self.length = len(self.videos_names)
        
        self.downscale_coef = downscale_coef
        self.video_processor = VideoProcessor(vae_scale_factor=downscale_coef)
        self.height = 480
        self.width = 832

        self.stage_2_training = stage_2_training
        self.sample_type = sample_type
        self.rng = None if seed is None else torch.Generator().manual_seed(seed)
        
    def __len__(self):
        return self.length

    @staticmethod
    def mask_frame_dropout_one(
        mask: torch.Tensor,                 # [C, T, H, W]
        *,
        fps: int = 8,
        early_seconds: int = 2,             # 2s -> 16 frames at 8 fps
        early_mass: float = 0.60,           # 60% mass on the early block
        black_value: float = -1.0,
        uniform: bool = False,              # set True for uniform selection
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Applies the non-uniform frame dropout to a single sample mask [C,T,H,W]."""
        assert mask.ndim == 4, "Expected [C, T, H, W]"
        C, T, H, W = mask.shape
        if T <= 1:
            return mask

        # candidates exclude first frame (t=0)
        Tm1 = T - 1
        early = min(fps * early_seconds, Tm1)

        # build probabilities over {1..T-1}
        if uniform or early == 0 or early == Tm1:
            probs = torch.full((Tm1,), 1.0 / Tm1, dtype=torch.float32)
        else:
            probs = torch.empty(Tm1, dtype=torch.float32)
            probs[:early]  = early_mass / early                 # 60% across first `early`
            probs[early:]  = (1.0 - early_mass) / (Tm1 - early) # 40% across the rest
            probs /= probs.sum()                                # safety

        # sample dropout ratio r \in {0.1, ..., 1.0}
        steps = torch.arange(1, 11, dtype=torch.float32) / 10.0
        r = steps[torch.randint(0, len(steps), (1,), generator=generator)].item()
        num_to_drop = int(Tm1 * r)
        if num_to_drop == 0:
            return mask

        # sample frames without replacement (never includes t=0)
        rel_idx = torch.multinomial(probs, num_samples=num_to_drop,
                                    replacement=False, generator=generator)  # [n]
        frame_ids = (rel_idx + 1).long()  # map {0..Tm1-1} -> {1..T-1}

        out = mask.clone()
        out[:, frame_ids, :, :] = black_value
        return out

    
    @staticmethod
    def resize_for_crop(image, crop_h, crop_w):
        img_h, img_w = image.shape[-2:]
        if img_h >= crop_h and img_w >= crop_w:
            coef = max(crop_h / img_h, crop_w / img_w)
        elif img_h <= crop_h and img_w <= crop_w:
            coef = max(crop_h / img_h, crop_w / img_w)
        else:
            coef = crop_h / img_h if crop_h > img_h else crop_w / img_w 
        out_h, out_w = int(img_h * coef), int(img_w * coef)
        resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
        return resized_image


    def prepare_frames(self, input_images, video_size, do_resize=True, do_crop=True):
        input_images = np.stack([np.array(x) for x in input_images])
        images_tensor = torch.from_numpy(input_images).permute(0, 3, 1, 2) / 127.5 - 1
        if do_resize:
            images_tensor = [self.resize_for_crop(x, crop_h=video_size[0], crop_w=video_size[1]) for x in images_tensor]
        if do_crop:
            images_tensor = [transforms.functional.center_crop(x, video_size) for x in images_tensor]
        if isinstance(images_tensor, list):
            images_tensor = torch.stack(images_tensor)
        return images_tensor.unsqueeze(0)
    

    def prepare_controlnet_frames(self, controlnet_frames, height, width, dtype):
        prepared_frames = self.prepare_frames(controlnet_frames, (height, width))
        controlnet_encoded_frames = prepared_frames.to(dtype=dtype)
        return controlnet_encoded_frames.permute(0, 2, 1, 3, 4).contiguous()


    def get_batch(self, idx):
        video_name = self.videos_names[idx]

        text_embeds_path = os.path.join(self.text_embeds_dir, video_name.replace(".mp4", ".pt").replace('masks', 'main'))
        text_embeds = torch.load(text_embeds_path, map_location="cpu", weights_only=True)

        latents_path = os.path.join(self.latents_dir, video_name.replace(".mp4", ".pt").replace('masks', 'main'))
        latents = torch.load(latents_path, map_location="cpu", weights_only=True)

        image_path = os.path.join(self.control_video_dir, video_name.replace(".mp4", ".pt").replace('masks', 'main'))
        image = torch.load(image_path, map_location="cpu", weights_only=True)

        video_path = os.path.join(self.controlnet_video_dir, video_name)
        
        img_h, img_w = latents.shape[-2:]
        controlnet_video = load_video(video_path)
        controlnet_video = self.prepare_controlnet_frames(
            controlnet_frames=controlnet_video,
            height=img_h * self.downscale_coef,
            width=img_w * self.downscale_coef,
            dtype=torch.bfloat16
            )
        if self.stage_2_training:
            controlnet_video = self.mask_frame_dropout_one(
                controlnet_video.squeeze(0),
                fps=8,
                early_seconds=2,
                early_mass=0.60,
                black_value=-1.0,
                uniform=self.sample_type=='uniform',              # set True for uniform strategy
                generator=self.rng,
            ).unsqueeze(0)


        return latents, image, text_embeds, controlnet_video
        
    def __getitem__(self, idx):
        while True:
            try:
                latents, latent_condition, text_embeds, controlnet_video = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)
        data = {
            'latents': latents[0],  ## [C, F, H, W] torch.Size([16, 21, 60, 104])
            'text_embeds': text_embeds[0],
            'latent_condition': latent_condition,
            'controlnet_video': controlnet_video   ## [C, F, H, W] torch.Size([3, 81, 480, 832])
        }
        return data
    