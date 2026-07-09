# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""Visualization utilities for colorized depth maps and videos."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

from .logging import get_local_logger

logger = get_local_logger(__name__)


def colorize_depth(
    depth: torch.Tensor,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "Spectral",
) -> np.ndarray:
    """
    Colorize a depth map using a matplotlib colormap.

    Args:
        depth: [H, W] or [N, H, W] depth tensor
        vmin: min depth for normalization (default: 2nd percentile)
        vmax: max depth for normalization (default: 98th percentile)
        cmap: matplotlib colormap name (default: "turbo")

    Returns:
        RGB uint8 array of shape [H, W, 3] or [N, H, W, 3]
    """
    depth_np = depth.detach().cpu().float().numpy()
    is_batched = depth_np.ndim == 3

    if not is_batched:
        depth_np = depth_np[None]

    valid = depth_np[depth_np > 0]
    if vmin is None:
        vmin = float(np.percentile(valid, 2)) if valid.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.percentile(valid, 98)) if valid.size > 0 else 1.0

    # Normalize to [0, 1]
    normed = (depth_np - vmin) / max(vmax - vmin, 1e-6)
    normed = np.clip(normed, 0, 1)

    # Apply matplotlib colormap (returns RGBA float in [0, 1])
    colormap = matplotlib.colormaps[cmap]
    colored = colormap(normed)[..., :3]  # drop alpha -> [N, H, W, 3]
    colored = (colored * 255).astype(np.uint8)

    if not is_batched:
        colored = colored[0]

    return colored


def save_depth_frames(
    depth: torch.Tensor,
    output_dir: str,
    prefix: str = "frame",
    vmin: float = None,
    vmax: float = None,
    cmap: str = "turbo",
) -> list[Path]:
    """
    Save colorized depth maps as individual PNG frames.

    Args:
        depth: [N, H, W] depth tensor
        output_dir: directory to save frames
        prefix: filename prefix
        vmin, vmax: depth range for color normalization
        cmap: matplotlib colormap name

    Returns:
        List of saved file paths
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    colored = colorize_depth(depth, vmin=vmin, vmax=vmax, cmap=cmap)  # [N, H, W, 3]
    paths = []
    for i in range(colored.shape[0]):
        path = out / f"{prefix}_{i:04d}.png"
        Image.fromarray(colored[i]).save(path)
        paths.append(path)
    return paths


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _save_video_ffmpeg(frames: list[Image.Image], output_path: Path, fps: int):
    """Encode frames to MP4 using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames):
            frame.save(f"{tmpdir}/frame_{i:04d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            f"{tmpdir}/frame_%04d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)


def _save_video_gif(frames: list[Image.Image], output_path: Path, fps: int):
    """Fallback: save as GIF when ffmpeg is unavailable."""
    output_path = output_path.with_suffix(".gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return output_path


def encode_video(frames: list[Image.Image], output_path: Path, fps: int) -> Path:
    """Encode frames to MP4 (or GIF if ffmpeg unavailable). Returns the saved path."""
    if _has_ffmpeg():
        out = output_path.with_suffix(".mp4")
        _save_video_ffmpeg(frames, out, fps)
        return out
    else:
        out = _save_video_gif(frames, output_path, fps)
        logger.info("ffmpeg not found, saved as GIF instead")
        return out


def save_depth_vis(
    depth: torch.Tensor,
    output_path: str,
    fps: int = 10,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "Spectral",
) -> Path:
    """
    Save colorized depth as PNG (single frame) or MP4/GIF (multiple frames).

    Args:
        depth: [N, H, W] depth tensor
        output_path: base path; suffix is overridden to .png or .mp4/.gif
        fps: frames per second (multi-frame only)
        vmin, vmax: depth range for color normalization
        cmap: matplotlib colormap name

    Returns:
        Path to the saved file
    """
    colored = colorize_depth(depth, vmin=vmin, vmax=vmax, cmap=cmap)  # [N, H, W, 3]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if colored.shape[0] == 1:
        out = output_path.with_suffix(".png")
        Image.fromarray(colored[0]).save(out)
        return out

    frames = [Image.fromarray(colored[i]) for i in range(colored.shape[0])]
    return encode_video(frames, output_path, fps)


def save_side_by_side_vis(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    output_path: str,
    fps: int = 10,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "Spectral",
) -> Path:
    """
    Save side-by-side RGB + depth as PNG (single frame) or MP4/GIF (multiple frames).

    Args:
        rgb: [N, 3, H, W] in [0, 1]
        depth: [N, H, W]
        output_path: base path; suffix is overridden to .png or .mp4/.gif
        fps: frames per second (multi-frame only)
        vmin, vmax: depth range for color normalization
        cmap: matplotlib colormap name

    Returns:
        Path to the saved file
    """
    rgb_np = (
        (rgb.detach().cpu().float().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    depth_colored = colorize_depth(depth, vmin=vmin, vmax=vmax, cmap=cmap)  # [N, H, W, 3]
    combined = np.concatenate([rgb_np, depth_colored], axis=2)  # [N, H, 2*W, 3]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if combined.shape[0] == 1:
        out = output_path.with_suffix(".png")
        Image.fromarray(combined[0]).save(out)
        return out

    # Ensure even dimensions (required by libx264)
    N, H, W2, _ = combined.shape
    if H % 2 != 0 or W2 % 2 != 0:
        combined = combined[:, : H & ~1, : W2 & ~1, :]

    frames = [Image.fromarray(combined[i]) for i in range(combined.shape[0])]
    return encode_video(frames, output_path, fps)
