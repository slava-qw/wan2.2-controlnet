import os
import glob
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from decord import VideoReader
from controlnet_aux import CannyDetector, HEDdetector, MidasDetector

import torch
import pandas as pd


def log(rank, *msg):
    print(f"[rank {rank}]", *msg, flush=True)


def resolve_rank_and_device(args):
    # TorchRun / Slurm set these reliably on ROCm
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            log(local_rank, "CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return local_rank, world_size, device


def init_controlnet(controlnet_type):
    if controlnet_type in ['canny']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators')


def save_video(out_path, frames, fps, desired_size=None):
    # frames: list/array of HxWxC (uint8/RGB or BGR), we write BGR
    if desired_size is not None:
        w, h = desired_size
        frames = [cv2.resize(np.array(f), (w, h), interpolation=cv2.INTER_AREA) for f in frames]

    # Ensure BGR for OpenCV
    bgr_frames = []
    for f in frames:
        arr = np.array(f)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # assume RGB -> BGR
            arr = arr[:, :, ::-1]
        bgr_frames.append(arr)

    img_h, img_w = bgr_frames[0].shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (img_w, img_h))
    for frame in bgr_frames:
        writer.write(frame)
    writer.release()


controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
    'depth': MidasDetector,
}


def process_partition(rank, world_size, device, args):
    # Diagnostics
    log(rank, f"torch.cuda.is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} "
              f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
              f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")

    df = pd.read_csv(args.csv_path)
    if "video" not in df.columns or "ids" not in df.columns:
        raise RuntimeError("CSV must contain 'video' and 'ids' columns")

    # input_video_paths = [os.path.join(args.input_video_dir, vid_path.replace('main', 'masks'))
                        #  for vid_path in df['video'].tolist()]
    input_video_paths = [os.path.join(args.input_video_dir, vid_path)
                         for vid_path in df['vel_masks'].tolist()]
    total_items = len(input_video_paths)

    # Partition (strided) across WORLD_SIZE
    local_indices = list(range(rank, total_items, world_size))
    log(rank, f"Assigned {len(local_indices)}/{total_items}. Sample idx: {local_indices[:6]}")

    # Skip already produced outputs by checking the OUT directory contents
    existing_outputs = set(os.path.basename(p) for p in glob.glob(os.path.join(args.out_controlnet_video_dir, "*.mp4")))

    controlnet_model = None
    if args.apply_controlnet:
        controlnet_model = init_controlnet(args.controlnet_type)
        log(rank, f"Loaded ControlNet annotator: {args.controlnet_type}")

    desired_size = None
    if args.width > 0 and args.height > 0:
        desired_size = (args.width, args.height)

    for gi in tqdm(local_indices, desc=f"worker-{rank}", total=len(local_indices), position=rank, leave=False):
        input_video_path = input_video_paths[gi]
        vid_id = df.iloc[gi]['ids']

        basename = os.path.basename(input_video_path)
        # save_basename = f"{basename.split('_')[0]}_{vid_id}.mp4"
        save_basename = f"{basename.split('.mp4')[0]}_{vid_id}.mp4"

        out_path = os.path.join(args.out_controlnet_video_dir, save_basename)

        if (not args.overwrite) and (save_basename in existing_outputs or os.path.exists(out_path)):
            continue

        # Keep dataset id ordering assertion (using global index)
        assert vid_id == gi, f"Wrong order of videos and ids; got gi={gi}, vid_id={vid_id}"

        try:
            vr = VideoReader(input_video_path)
            video_length = len(vr)
            fps_original = float(vr.get_avg_fps()) if vr.get_avg_fps() is not None else 24.0

            clip_length = min(video_length, (args.sample_n_frames - 1) * args.sample_stride + 1)
            batch_index = np.linspace(0, clip_length - 1, args.sample_n_frames, dtype=int)
            np_video = vr.get_batch(batch_index).asnumpy()
            del vr

            if args.apply_controlnet and controlnet_model is not None:
                # Apply detector per frame
                frames = [controlnet_model(x) for x in np_video]
            else:
                frames = np_video

            save_video(out_path, frames, fps_original, desired_size=desired_size)

        except Exception as e:
            log(rank, f"Error on global index {gi} ({input_video_path}): {e}")


def main(args):
    rank, world_size, device = resolve_rank_and_device(args)
    log(rank, f"LOCAL_RANK={rank} WORLD_SIZE={world_size} controlnet={args.controlnet_type} apply={args.apply_controlnet}")

    os.makedirs(args.out_controlnet_video_dir, exist_ok=True)
    process_partition(rank, world_size, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ControlNet conditioning videos (torchrun multi-GPU, ROCm/Slurm friendly).")
    parser.add_argument("--input_video_dir", type=str, required=True, help="Directory with videos for processing")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with at least 'video' and 'ids' columns")
    parser.add_argument("--out_controlnet_video_dir", type=str, required=True, help="Directory for output controlnet videos (.mp4)")
    parser.add_argument("--controlnet_type", type=str, default="canny", choices=["canny","hed","depth"],
                        help="Annotator type to apply if --apply_controlnet is given")
    parser.add_argument("--apply_controlnet", action="store_true",
                        help="If set, runs the chosen annotator; otherwise uses the raw frames (default).")
    parser.add_argument("--sample_stride", type=int, default=2, help="Get each Nth frame")
    parser.add_argument("--width", type=int, default=832, help="Resize width for output (set <=0 to keep original)")
    parser.add_argument("--height", type=int, default=480, help="Resize height for output (set <=0 to keep original)")
    parser.add_argument("--sample_n_frames", type=int, default=81, help="Total frames per clip")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for any annotator ops")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing .mp4 instead of skipping")
    args = parser.parse_args()
    main(args)
