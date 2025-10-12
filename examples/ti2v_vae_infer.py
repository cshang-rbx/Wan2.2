#!/usr/bin/env python3
"""
Minimal example for running the Wan2.2 TI2V-5B VAE.

This script loads a video file, encodes it with the VAE shipped with TI2V-5B,
decodes the latent back to pixel space, and saves the reconstruction.
"""

import argparse
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

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


def _make_stride_compatible(video: torch.Tensor,
                            t_stride: int,
                            h_stride: int,
                            w_stride: int,
                            mode: str) -> torch.Tensor:
    """Adjust temporal and spatial sizes to satisfy stride constraints."""
    c, t, h, w = video.shape

    def _trim(value: int, stride: int, offset: int = 0) -> int:
        trimmed = (value - offset) // stride * stride + offset
        return max(trimmed, stride + offset)

    def _pad(value: int, stride: int, offset: int = 0) -> int:
        padded = math.ceil((value - offset) / stride) * stride + offset
        return max(padded, stride + offset)

    need_temporal_fix = (t - 1) % t_stride != 0
    need_h_fix = h % h_stride != 0
    need_w_fix = w % w_stride != 0

    if mode == "error":
        if need_temporal_fix:
            raise ValueError(
                f"Video length {t} is not compatible with the temporal stride {t_stride}. "
                f"Please trim/pad the clip so that (frames - 1) is divisible by {t_stride}.")
        if need_h_fix or need_w_fix:
            raise ValueError(
                f"Spatial size {(h, w)} must be divisible by ({h_stride}, {w_stride}).")
        return video

    if mode not in {"trim", "pad"}:
        raise ValueError(f"Unsupported stride_compat option '{mode}'.")

    new_t = t
    new_h = h
    new_w = w
    if need_temporal_fix:
        new_t = (_trim if mode == "trim" else _pad)(t, t_stride, offset=1)
    if need_h_fix:
        new_h = (_trim if mode == "trim" else _pad)(h, h_stride)
    if need_w_fix:
        new_w = (_trim if mode == "trim" else _pad)(w, w_stride)

    if mode == "trim":
        video = video[:, :new_t, :new_h, :new_w]
    else:
        pad_t = new_t - t
        pad_h = new_h - h
        pad_w = new_w - w
        if pad_t > 0:
            last = video[:, -1:].repeat(1, pad_t, 1, 1)
            video = torch.cat([video, last], dim=1)
        if pad_h > 0 or pad_w > 0:
            # Replication padding on very large tensors can overflow CUDA's 32-bit indexing,
            # so perform the padding on CPU before moving back to the target device.
            pad_args = (0, pad_w, 0, pad_h)
            target_device = video.device
            video = F.pad(video.cpu(), pad_args, mode="replicate").to(target_device)

    return video


def run_inference(ckpt_dir: str,
                  video_path: str,
                  output_path: str,
                  device: torch.device,
                  fps: Optional[int] = None,
                  stride_compat: str = "error") -> None:
    cfg = WAN_CONFIGS["ti2v-5B"]
    vae_path = os.path.join(ckpt_dir, cfg.vae_checkpoint)
    if not os.path.exists(vae_path):
        raise FileNotFoundError(
            f"Could not find VAE checkpoint at '{vae_path}'. "
            "Make sure --ckpt_dir points to the extracted TI2V-5B weights.")

    video = _load_video(video_path).to(device)
    t_stride, h_stride, w_stride = cfg.vae_stride
    video = _make_stride_compatible(video, t_stride, h_stride, w_stride, stride_compat)

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
    parser.add_argument(
        "--stride-compat",
        choices=["error", "trim", "pad"],
        default="pad",
        help="How to handle temporal/spatial sizes that do not match the VAE stride.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    run_inference(
        ckpt_dir=args.ckpt_dir,
        video_path=args.video,
        output_path=args.output,
        device=device,
        stride_compat=args.stride_compat)


if __name__ == "__main__":
    main()
