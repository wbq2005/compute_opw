# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
CAPA: depth Completion As Parameter-efficient Adaptation.

Usage:
    python run.py --config config/vggt_lora.yaml --input input/sample_data/scannet_sift_noise_10pct/scene0806_00_framee81c664807_view5e955e0879.pt
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from capa import CAPAProtocol
from capa.utils.logging import get_local_logger
from capa.utils.metric import compute_depth_metrics, format_metrics
from capa.utils.visualize import save_depth_vis, save_side_by_side_vis

logger = get_local_logger("capa.run")


def load_sample(path: str) -> dict:
    """Load a sample .pt file containing scene data."""
    data = torch.load(path, weights_only=False, map_location="cpu")
    logger.info(
        f"Loaded {path}: "
        f"rgb={data['rgb_nv3hw'].shape}, "
        f"depth_cond={data['depth_condition_nvhw'].shape}, "
        f"mask={data['mask_condition_nvhw'].shape}"
    )
    return data


def main():
    parser = argparse.ArgumentParser(description="CAPA depth completion")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input .pt file or directory",
    )
    parser.add_argument("--output", "-o", type=str, default="output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save colorized depth visualizations",
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS for depth video output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("capa").setLevel(logging.DEBUG)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {args.config} -> {config['model_name']}+{config['tuning_mode']}")

    # Setup device
    device = torch.device(args.device)

    # Initialize protocol
    protocol = CAPAProtocol(config, device)

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_dir():
        pt_files = sorted(input_path.glob("**/*.pt"))
    else:
        pt_files = [input_path]
    logger.info(f"Found {len(pt_files)} input file(s)")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each scene
    for pt_file in pt_files:
        logger.info(f"Processing: {pt_file.stem}")
        data = load_sample(str(pt_file))

        rgb_n3hw = data["rgb_nv3hw"]
        depth_cond_nhw = data["depth_condition_nvhw"]
        mask_cond_nhw = data["mask_condition_nvhw"]

        # Run CAPA
        depth_pred_nhw = protocol.run(rgb_n3hw, depth_cond_nhw, mask_cond_nhw)

        # Evaluate if GT available
        vmax = None
        if "depth_gt_nvhw" in data:
            depth_gt = data["depth_gt_nvhw"].to(device)
            mask_gt = depth_gt > 0
            metrics = compute_depth_metrics(depth_pred_nhw, depth_gt, mask_gt)
            logger.info(f"Metrics: {format_metrics(metrics)}")
            valid_gt = depth_gt[mask_gt]
            _, vmax = float(valid_gt.min()), float(valid_gt.max())

        # Save prediction tensor
        save_path = output_dir / f"{pt_file.stem}_pred.pt"
        torch.save({"depth_pred_nhw": depth_pred_nhw.cpu()}, save_path)
        logger.info(f"Depth saved to: {save_path}")

        # Save colorized depth visualization (PNG for single frame, MP4/GIF otherwise)
        if args.save_vis:
            depth_vis = save_depth_vis(
                depth_pred_nhw,
                str(output_dir / f"{pt_file.stem}_depth"),
                fps=args.fps,
                vmin=0.0,
                vmax=vmax,
            )
            logger.info(f"Depth vis saved to: {depth_vis}")
            sbs_vis = save_side_by_side_vis(
                rgb_n3hw,
                depth_pred_nhw,
                str(output_dir / f"{pt_file.stem}_sidebyside"),
                fps=args.fps,
                vmin=0.0,
                vmax=vmax,
            )
            logger.info(f"Side-by-side vis saved to: {sbs_vis}")

        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
