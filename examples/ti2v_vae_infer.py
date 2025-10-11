#!/usr/bin/env python3
"""
Minimal example for running the Wan2.2 TI2V-5B VAE.

This script loads a video file, encodes it with the VAE shipped with TI2V-5B,
decodes the latent back to pixel space, and saves the reconstruction.
"""

import argparse
import os
from typing import Optional

import torch

from wan.configs import WAN_CONFIGS
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.utils.utils import save_video

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - imageio.v3 is preferred but optional
    import imageio as iio  # type: ignore


def _load_video(path: str) -> torch.Tensor:
    """Load a video file and return a tensor in (C, T, H, W) with values in [-1, 1]."""
    frames = iio.imread(path, index=None)  # (T, H, W, C), dtype=uint8
    if frames.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, got shape {frames.shape}.")
    if frames.shape[-1] != 3:
        raise ValueError("Only RGB videos are supported.")
    tensor = torch.from_numpy(frames).float() / 127.5 - 1.0  # map to [-1, 1]
    return tensor.permute(3, 0, 1, 2)  # (C, T, H, W)


def _save_video(video: torch.Tensor, path: str, fps: int) -> None:
    """Save a (C, T, H, W) tensor to disk using Wan's helper."""
    save_video(video.unsqueeze(0), save_file=path, fps=fps, nrow=1, normalize=True)


def run_inference(ckpt_dir: str,
                  video_path: str,
                  output_path: str,
                  device: torch.device,
                  fps: Optional[int] = None) -> None:
    cfg = WAN_CONFIGS["ti2v-5B"]
    vae_path = os.path.join(ckpt_dir, cfg.vae_checkpoint)
    if not os.path.exists(vae_path):
        raise FileNotFoundError(
            f"Could not find VAE checkpoint at '{vae_path}'. "
            "Make sure --ckpt_dir points to the extracted TI2V-5B weights.")

    video = _load_video(video_path).to(device)
    t_stride, h_stride, w_stride = cfg.vae_stride
    if (video.shape[1] - 1) % t_stride != 0:
        raise ValueError(
            f"Video length {video.shape[1]} is not compatible with the temporal stride "
            f"{t_stride}. Please trim/pad the clip so that (frames - 1) is divisible by {t_stride}.")
    if video.shape[-2] % h_stride != 0 or video.shape[-1] % w_stride != 0:
        raise ValueError(
            f"Spatial size {(video.shape[-2], video.shape[-1])} must be divisible by "
            f"({h_stride}, {w_stride}).")

    vae = Wan2_2_VAE(vae_pth=vae_path, device=device)
    vae.model = vae.model.to(device)

    with torch.inference_mode():
        latents = vae.encode([video])[0]
        recon = vae.decode([latents])[0].cpu()

    _save_video(recon, output_path, fps or cfg.sample_fps)

    print(f"Input video shape: {tuple(video.shape)}")
    print(f"Latent shape: {tuple(latents.shape)}")
    print(f"Reconstruction saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan2.2 TI2V-5B VAE inference.")
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Directory containing the TI2V-5B checkpoints (expects Wan2.2_VAE.pth).")
    parser.add_argument(
        "--video",
        default="examples/wan_animate/animate/video.mp4",
        help="Input video to encode and decode. Defaults to the sample clip.")
    parser.add_argument(
        "--output",
        default="ti2v_vae_reconstruction.mp4",
        help="Where to write the reconstructed video.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    run_inference(
        ckpt_dir=args.ckpt_dir,
        video_path=args.video,
        output_path=args.output,
        device=device)


if __name__ == "__main__":
    main()
