# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
CAPA: depth Completion As Parameter-efficient Adaptation.

Usage:
    # High-resolution baseline: cache predictions without loading GMFlow, then
    # compute OPW with scripts/audit_opw_metric.py.
    CUDA_VISIBLE_DEVICES=7 python run.py \
        --config config/vggt_baseline.yaml \
        --input dataset/metropolis/metropolis_8line_noisy_v3 \
        --output output/noise_probe/metropolis_8line_v3/vggt \
        --save-pt --no-opw

    # Multi-GPU (auto-detected from CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/vggt_vpt.yaml --input dataset/scannet/scannet_lt3m_noisy
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

from capa import CAPAProtocol
from capa.utils.logging import get_local_logger
from capa.utils.metric import (
    OPW_PROTOCOL,
    average_metrics,
    compute_depth_metrics,
    compute_opw,
    format_metrics,
    load_gmflow,
    resolve_gmflow_paths,
)
from capa.utils.visualize import save_depth_vis, save_side_by_side_vis

logger = get_local_logger("capa.run")

_PROJECT_ROOT = Path(__file__).resolve().parent


def _json_safe(value):
    """Return a JSON-standard representation without NaN or Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _build_opw_evaluation_metadata(
    *,
    online_opw: bool,
    avg_metrics: dict,
    per_sample: list[dict],
    gmflow_ckpt: str | None,
    gmflow_repo: str | None,
    beta: float,
    fb_consistency: bool,
    flow_batch_size: int | None,
    flow_max_side: int | None,
) -> dict:
    """Build the canonical OPW protocol record stored in summary.json."""
    finite_opw_count = sum(
        1
        for item in per_sample
        if isinstance((value := item.get("opw")), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    has_average = "opw" in avg_metrics
    if online_opw and has_average != bool(finite_opw_count):
        raise RuntimeError(
            "Online OPW summary is inconsistent: avg_metrics.opw presence does "
            f"not match {finite_opw_count} finite per-sample values"
        )
    if not online_opw:
        status = "not_requested"
    elif finite_opw_count:
        status = "computed_online"
    else:
        status = "not_applicable"

    return {
        "status": status,
        "online": online_opw,
        "source_json": None,
        "mean_opw": avg_metrics.get("opw") if online_opw else None,
        "num_valid_scenes": finite_opw_count if online_opw else None,
        "num_total_scenes": finite_opw_count if online_opw else None,
        "gmflow_ckpt": gmflow_ckpt if online_opw else None,
        "gmflow_repo": gmflow_repo if online_opw else None,
        "beta": beta if online_opw else None,
        "fb_consistency": fb_consistency if online_opw else None,
        "protocol": OPW_PROTOCOL if online_opw else None,
        "depth_key": "depth_pred_nhw" if online_opw else None,
        "eval_mask_key": "depth_gt_nvhw" if online_opw else None,
        "flow_batch_size": flow_batch_size if online_opw else None,
        "flow_max_side": flow_max_side if online_opw else None,
    }


def _set_global_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch CPU+CUDA RNGs for reproducibility.

    Called once at process startup AND before each sample, so that a sample
    processed alone vs as the N-th sample in a batch sees the same RNG state.
    Note: we do NOT enable cudnn.deterministic here, because the run-to-run
    drift is dominated by RNG (not cuDNN algo selection) for this pipeline,
    and full determinism would noticeably slow down training/inference.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("CAPA_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_path(stored_path: str) -> str:
    """Resolve a path stored in .pt files.

    Paths are stored as relative (e.g. 'dataset/scannet/scannet_base/scene0707_00.pt').
    If not absolute, resolve relative to _PROJECT_ROOT.
    """
    p = Path(stored_path)
    if p.is_absolute():
        if p.exists():
            return str(p)
        # Try interpreting as relative from "dataset/" onwards
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "dataset":
                local = _PROJECT_ROOT / Path(*parts[i:])
                if local.exists():
                    return str(local)
                break
        raise FileNotFoundError(f"Path not found: {stored_path}")
    # Relative path: resolve from project root
    resolved = _PROJECT_ROOT / p
    if resolved.exists():
        return str(resolved)
    raise FileNotFoundError(
        f"Path not found: {stored_path}\n"
        f"Resolved to: {resolved}"
    )


def _load_scannet_coo(cond_data: dict) -> dict:
    """Decode ScanNet COO sparse .pt (source_base + conditions).

    source_base points to a self-contained base .pt with {rgb_nv3hw, depth_gt_nvhw}.
    """
    base_path = _resolve_path(cond_data["source_base"])
    base_data = torch.load(base_path, weights_only=False, map_location="cpu")
    rgb = base_data["rgb_nv3hw"]
    depth_gt = base_data["depth_gt_nvhw"]
    H, W = depth_gt.shape[1], depth_gt.shape[2]

    cond_list = cond_data["conditions"]
    sparse_list, mask_list = [], []
    for comp in cond_list:
        sparse = torch.zeros(H, W, dtype=torch.float32)
        rows, cols, vals = comp["rows"].long(), comp["cols"].long(), comp["vals"]
        if len(rows) > 0:
            sparse[rows, cols] = vals
        sparse_list.append(sparse)
        mask_list.append(sparse > 0)

    return {
        "rgb_nv3hw": rgb,
        "depth_gt_nvhw": depth_gt,
        "depth_condition_nvhw": torch.stack(sparse_list),
        "mask_condition_nvhw": torch.stack(mask_list),
    }


def load_sample(path: str) -> dict:
    """Load a sample .pt — supports standard, source_pt ref, COO sparse, and ScanNet compact formats."""
    data = torch.load(path, weights_only=False, map_location="cpu")

    # Format 1: standard self-contained (has rgb_nv3hw directly)
    if "rgb_nv3hw" in data:
        logger.debug(f"Loaded {path}: rgb={data['rgb_nv3hw'].shape}")
        return data

    # Format 3: COO sparse with source_pt (Metropolis/7scenes lightweight)
    # Must check BEFORE Format 2 since both have "source_pt"
    if "conditions" in data and "source_pt" in data:
        source_path = _resolve_path(data["source_pt"])
        source_data = torch.load(source_path, weights_only=False, map_location="cpu")
        rgb = source_data["rgb_nv3hw"]
        depth_gt = source_data["depth_gt_nvhw"]
        H, W = depth_gt.shape[1], depth_gt.shape[2]

        cond_list = data["conditions"]
        sparse_list, mask_list = [], []
        for coo in cond_list:
            sparse = torch.zeros(H, W, dtype=torch.float32)
            rows = coo["rows"].long()
            cols = coo["cols"].long()
            vals = coo["vals"]
            if len(rows) > 0:
                sparse[rows, cols] = vals
            sparse_list.append(sparse)
            mask_list.append(sparse > 0)

        decoded = {
            "rgb_nv3hw": rgb,
            "depth_gt_nvhw": depth_gt,
            "depth_condition_nvhw": torch.stack(sparse_list),
            "mask_condition_nvhw": torch.stack(mask_list),
        }
        logger.debug(f"Loaded {path} (COO+source_pt): rgb={rgb.shape}, cond_pts={sum(len(c['rows']) for c in cond_list)}")
        return decoded

    # Format 2: lightweight with source_pt reference (dense condition + source_pt)
    if "source_pt" in data:
        source_path = _resolve_path(data["source_pt"])
        source_data = torch.load(source_path, weights_only=False, map_location="cpu")
        data["rgb_nv3hw"] = source_data["rgb_nv3hw"]
        data["depth_gt_nvhw"] = source_data["depth_gt_nvhw"]
        logger.debug(f"Loaded {path} (source_pt ref): rgb={data['rgb_nv3hw'].shape}")
        return data

    # Format 4: ScanNet COO sparse (source_base + conditions)
    if "source_base" in data and "conditions" in data:
        decoded = _load_scannet_coo(data)
        logger.debug(f"Loaded {path} (COO+source_base): rgb={decoded['rgb_nv3hw'].shape}")
        return decoded

    raise ValueError(f"Unknown .pt format in {path}, keys: {list(data.keys())}")


def process_samples(
    rank: int,
    pt_files: list[Path],
    config: dict,
    output_dir: Path,
    device: torch.device,
    save_vis: bool,
    save_pt: bool,
    fps: int,
    result_file: Path | None = None,
    online_opw: bool = False,
    opw_beta: float = 50.0,
    opw_fb_consistency: bool = False,
    opw_flow_batch_size: int | None = None,
    opw_flow_max_side: int | None = None,
    save_trace: bool = False,
) -> tuple[list[dict], list[float]]:
    """
    Worker function: process a subset of samples on a single GPU.

    Args:
        rank: worker index (for logging / progress bar position)
        pt_files: list of .pt files assigned to this worker
        config: YAML config dict
        output_dir: where to save predictions
        device: torch.device for this worker
        save_vis: whether to save visualizations
        save_pt: whether to save per-sample _pred.pt files
        fps: FPS for video output
        result_file: if provided, write (metrics, times) to this file for cross-process collection
        online_opw: compute OPW inside this run. Offline audit is preferred for
            high-resolution clips because VGGT and GMFlow otherwise share VRAM.
        opw_beta: RGB visibility-weight beta.
        opw_fb_consistency: enable forward-backward flow consistency filtering.
        opw_flow_batch_size: maximum adjacent frame pairs per GMFlow call.
        opw_flow_max_side: optional GMFlow inference resize limit.
        save_trace: if True, write per-frame {scale, shift, abs rel} trace JSON for each sample

    Returns:
        (all_metrics, all_times) — only meaningful when called in single-GPU mode
    """
    protocol = CAPAProtocol(config, device)
    seed = int(config.get("seed", 42))

    # GMFlow is loaded lazily when online OPW reaches its first video sample.
    flow_model: torch.nn.Module | None = None

    all_metrics: list[dict] = []
    all_times: list[float] = []

    desc = f"[GPU {device.index if device.index is not None else 0}] {config['model_name']}+{config['tuning_mode']}"
    pbar = tqdm(pt_files, desc=desc, unit="scene", position=rank, leave=True)
    for pt_file in pbar:
        # Per-sample artifacts must never leak through Python locals when a
        # cached prediction follows a freshly evaluated sample.
        stream_stats = None
        alignment_trace = None
        oracle_audit = None
        save_path = output_dir / f"{pt_file.stem}_pred.pt"
        pbar.set_postfix_str(pt_file.stem, refresh=True)

        # Skip if cached result already exists (only when save_pt is enabled)
        if save_pt and save_path.exists():
            data = load_sample(str(pt_file))
            saved = torch.load(save_path, weights_only=False, map_location=device)
            depth_pred_nhw = saved["depth_pred_nhw"].to(device)
            rgb_n3hw = data["rgb_nv3hw"]
            elapsed = None
            # Restore cached alignment trace (if any) for --save-trace replay.
            alignment_trace = saved.get("alignment_trace") if isinstance(saved, dict) else None
        else:
            data = load_sample(str(pt_file))
            rgb_n3hw = data["rgb_nv3hw"]
            depth_cond_nhw = data["depth_condition_nvhw"]
            mask_cond_nhw = data["mask_condition_nvhw"]

            _set_global_seed(seed)

            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            protocol.trace_depth_gt_nhw = data.get("depth_gt_nvhw")
            depth_pred_nhw = protocol.run(rgb_n3hw, depth_cond_nhw, mask_cond_nhw)
            protocol.trace_depth_gt_nhw = None
            depth_pred_nhw = depth_pred_nhw.to(device)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            all_times.append(elapsed)

            # Capture streaming stats if available (set by run_streaming)
            stream_stats = getattr(protocol, "last_stream_stats", None)
            # Capture per-frame alignment trace (set by both run_streaming
            # and _run_optimization). Used by --save-trace.
            alignment_trace = getattr(protocol, "last_alignment_trace", None)
            oracle_audit = getattr(protocol, "last_oracle_audit", None)

            # Oracle audit is a separate compact tensor artifact rather than
            # JSON: it stores low-resolution per-step candidates and dense-GT
            # labels for offline analysis only. It has no online effect.
            if oracle_audit is not None:
                audit_path = output_dir / f"{pt_file.stem}_oracle_audit.pt"
                torch.save(oracle_audit, audit_path)
                logger.info(f"Oracle audit saved: {audit_path}")

            if save_pt:
                payload = {"depth_pred_nhw": depth_pred_nhw.cpu()}
                if stream_stats is not None:
                    payload["stream_stats"] = stream_stats
                if alignment_trace is not None:
                    payload["alignment_trace"] = alignment_trace
                if oracle_audit is not None:
                    payload["oracle_audit_path"] = str(audit_path.name)
                torch.save(payload, save_path)

        # Evaluate if GT available
        vmax = None
        if "depth_gt_nvhw" in data:
            depth_gt = data["depth_gt_nvhw"].to(device)
            mask_gt = depth_gt > 0
            metrics = compute_depth_metrics(depth_pred_nhw, depth_gt, mask_gt)
            metrics["sample"] = pt_file.stem
            if elapsed is not None:
                metrics["time_s"] = round(elapsed, 3)
                metrics["peak_vram_mb"] = round(peak_vram_mb, 1)

            # Streaming stats: per-sample early-exit / step counts.
            # When loaded from cache, stream_stats may be in the saved .pt.
            stats_for_metrics = locals().get("stream_stats")
            if stats_for_metrics is None and save_pt and save_path.exists():
                stats_for_metrics = saved.get("stream_stats") if "saved" in locals() else None
            if stats_for_metrics is not None:
                metrics["stream_early_exit_frames"] = int(stats_for_metrics["num_early_exit_frames"])
                metrics["stream_num_frames"] = int(stats_for_metrics["num_frames"])
                metrics["stream_early_exit_ratio"] = round(stats_for_metrics["early_exit_ratio"], 4)
                metrics["stream_total_steps"] = int(stats_for_metrics["total_steps"])
                metrics["stream_max_total_steps"] = int(stats_for_metrics["max_total_steps"])
                metrics["stream_step_saving_ratio"] = round(stats_for_metrics["step_saving_ratio"], 4)
                metrics["stream_steps_per_frame"] = stats_for_metrics.get("steps_per_frame")
                metrics["stream_early_exit_per_frame"] = stats_for_metrics.get("early_exit_per_frame")
                if "support_calibration_enabled" in stats_for_metrics:
                    metrics["stream_support_calibration_enabled"] = bool(
                        stats_for_metrics["support_calibration_enabled"]
                    )
                if "support_calibration_avg_risk" in stats_for_metrics:
                    metrics["stream_support_calibration_avg_risk"] = round(
                        float(stats_for_metrics["support_calibration_avg_risk"]), 6
                    )
                if "support_calibration_slots" in stats_for_metrics:
                    metrics["stream_support_calibration_slots"] = int(
                        stats_for_metrics["support_calibration_slots"]
                    )
                if "tbaa_loss_gate_enabled" in stats_for_metrics:
                    metrics["stream_tbaa_loss_gate_enabled"] = bool(
                        stats_for_metrics["tbaa_loss_gate_enabled"]
                    )
                if "tbaa_loss_gate_steps" in stats_for_metrics:
                    metrics["stream_tbaa_loss_gate_steps"] = int(
                        stats_for_metrics["tbaa_loss_gate_steps"]
                    )
            # OPW is opt-in. Missing or failed OPW must never be represented by
            # NaN in a successful summary.
            if online_opw and depth_pred_nhw.shape[0] >= 2:
                if flow_model is None:
                    try:
                        flow_model = load_gmflow(device=device)
                    except Exception as exc:
                        raise RuntimeError(
                            "Online OPW was requested but GMFlow could not be loaded. "
                            "Use --no-opw and scripts/audit_opw_metric.py for the "
                            "recommended offline workflow."
                        ) from exc
                    logger.info("GMFlow loaded for online OPW computation.")

                opw_val = float(
                    compute_opw(
                        depth_pred_nhw,
                        rgb_n3hw.to(device),
                        flow_model=flow_model,
                        beta=opw_beta,
                        fb_consistency=opw_fb_consistency,
                        eval_mask=torch.isfinite(depth_gt) & (depth_gt > 0),
                        flow_batch_size=opw_flow_batch_size,
                        flow_max_side=opw_flow_max_side,
                    )
                )
                if not math.isfinite(opw_val):
                    raise RuntimeError(
                        f"Online OPW returned a non-finite value for {pt_file.stem}: "
                        f"{opw_val!r}"
                    )
                metrics["opw"] = round(opw_val, 6)

            all_metrics.append(metrics)
            valid_gt = depth_gt[mask_gt]
            _, vmax = float(valid_gt.min()), float(valid_gt.max())

            postfix = {"absrel": f"{metrics['absrel']:.4f}"}
            if "opw" in metrics and metrics["opw"] == metrics["opw"]:
                postfix["opw"] = f"{metrics['opw']:.2f}"
            if elapsed is not None:
                postfix["t"] = f"{elapsed:.1f}s"
            pbar.set_postfix(postfix, refresh=True)

        if save_vis:
            save_depth_vis(
                depth_pred_nhw,
                str(output_dir / f"{pt_file.stem}_depth"),
                fps=fps, vmin=0.0, vmax=vmax,
            )
            save_side_by_side_vis(
                rgb_n3hw, depth_pred_nhw,
                str(output_dir / f"{pt_file.stem}_sidebyside"),
                fps=fps, vmin=0.0, vmax=vmax,
            )

        # Write per-frame alignment trace JSON (scale, shift, abs rel).
        # Merges protocol's alignment_trace (sc/sh per frame) with per-frame
        # abs rel computed against GT (if available).
        if save_trace and alignment_trace:
            trace_records = list(alignment_trace)
            # Compute per-frame abs rel against GT when available.
            if "depth_gt_nvhw" in data:
                depth_gt_trace = data["depth_gt_nvhw"].to(device)
                mask_gt_trace = depth_gt_trace > 0
                from capa.utils.metric import absrel as _absrel_fn
                for rec in trace_records:
                    fi = rec.get("frame")
                    if fi is None or fi < 0 or fi >= depth_pred_nhw.shape[0]:
                        rec["absrel"] = None
                        continue
                    ar = _absrel_fn(
                        depth_pred_nhw[fi:fi+1],
                        depth_gt_trace[fi:fi+1],
                        mask_gt_trace[fi:fi+1],
                    )
                    rec["absrel"] = float(ar.item())
                del depth_gt_trace, mask_gt_trace
            # Normalize field names so plotting is uniform across methods:
            #   sc  — final scale applied to this frame
            #   sh  — final shift applied to this frame
            for rec in trace_records:
                if "sc_final" in rec:
                    rec.setdefault("sc", rec["sc_final"])
                elif "sc_fused" in rec:
                    rec.setdefault("sc", rec["sc_fused"])
                if "sh_final" in rec:
                    rec.setdefault("sh", rec["sh_final"])
                elif "sh_fused" in rec:
                    rec.setdefault("sh", rec["sh_fused"])
            method_name = trace_records[0].get("method", "unknown") if trace_records else "unknown"
            trace_payload = {
                "sample": pt_file.stem,
                "method": method_name,
                "alignment": {
                    "type": config.get("alignment", {}).get("type", "lsq"),
                    "n_points": config.get("alignment", {}).get("n_points", 12000),
                },
                "alignment_trace": trace_records,
            }
            trace_path = output_dir / f"{pt_file.stem}_trace.json"
            with trace_path.open("w") as f:
                json.dump(_json_safe(trace_payload), f, indent=2, allow_nan=False)
            logger.info(f"Trace saved: {trace_path} ({len(trace_records)} frames)")

        # Explicit GC + cache clear between scenes to prevent VRAM accumulation.
        # A single scene peaks at ~22.2 GB on a 24 GB GPU; without forced GC,
        # optimizer state and computation graphs from the previous scene can
        # outlive their Python references, causing OOM on the next scene.
        del data, depth_pred_nhw
        if "depth_gt" in dir():
            del depth_gt
        gc.collect()
        torch.cuda.empty_cache()

    # Save partial results for multi-GPU aggregation
    if result_file is not None:
        torch.save({"metrics": all_metrics, "times": all_times}, result_file)

    return all_metrics, all_times


def _worker_fn(
    rank: int,
    gpu_ids: list[int],
    all_pt_files: list[Path],
    config: dict,
    output_dir: Path,
    save_vis: bool,
    save_pt: bool,
    fps: int,
    tmp_dir: Path,
    online_opw: bool = False,
    opw_beta: float = 50.0,
    opw_fb_consistency: bool = False,
    opw_flow_batch_size: int | None = None,
    opw_flow_max_side: int | None = None,
    save_trace: bool = False,
):
    """Entry point for each spawned GPU worker process."""
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Partition files: round-robin assignment so each worker gets ~equal load
    my_files = [f for i, f in enumerate(all_pt_files) if i % len(gpu_ids) == rank]
    if not my_files:
        return

    result_file = tmp_dir / f"results_rank{rank}.pt"
    process_samples(
        rank=rank,
        pt_files=my_files,
        config=config,
        output_dir=output_dir,
        device=device,
        save_vis=save_vis,
        save_pt=save_pt,
        fps=fps,
        result_file=result_file,
        online_opw=online_opw,
        opw_beta=opw_beta,
        opw_fb_consistency=opw_fb_consistency,
        opw_flow_batch_size=opw_flow_batch_size,
        opw_flow_max_side=opw_flow_max_side,
        save_trace=save_trace,
    )


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
    parser.add_argument(
        "--no-opw",
        action="store_true",
        help=(
            "Skip online OPW. For high-resolution prediction caching, run the "
            "offline audit afterwards; summary omits OPW until that audit merges."
        ),
    )
    parser.add_argument(
        "--opw-fb-consistency",
        action="store_true",
        help="Enable forward-backward flow consistency filtering for online OPW.",
    )
    parser.add_argument(
        "--opw-flow-batch-size",
        type=int,
        default=2,
        help="Maximum adjacent frame pairs per online GMFlow call (default: 2).",
    )
    parser.add_argument(
        "--opw-flow-max-side",
        type=int,
        default=None,
        help="Optional maximum image side for online GMFlow inference.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save colorized depth visualizations",
    )
    parser.add_argument(
        "--save-pt",
        action="store_true",
        help="Save per-sample _pred.pt files (enables result caching; implied by --save-vis)",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save per-frame alignment trace JSON (scale, shift, abs rel) for each sample",
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS for depth video output")
    parser.add_argument(
        "--sample",
        "-s",
        type=int,
        default=None,
        help="Only process first N samples (for quick testing/debugging)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.opw_flow_batch_size is not None and args.opw_flow_batch_size <= 0:
        parser.error("--opw-flow-batch-size must be positive")
    if args.opw_flow_max_side is not None and args.opw_flow_max_side <= 0:
        parser.error("--opw-flow-max-side must be positive")

    online_opw = not args.no_opw
    opw_options_used = (
        args.opw_fb_consistency
        or args.opw_flow_batch_size != 2
        or args.opw_flow_max_side is not None
    )
    if opw_options_used and not online_opw:
        parser.error("Online OPW options cannot be combined with --no-opw")

    # --save-vis implies --save-pt (visualization needs cached predictions for replay)
    save_pt = args.save_pt or args.save_vis

    if args.verbose:
        logging.getLogger("capa").setLevel(logging.DEBUG)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.debug(f"Config: {args.config} -> {config['model_name']}+{config['tuning_mode']}")

    seed = int(config.get("seed", 42))
    _set_global_seed(seed)
    logger.info(f"Global seed set to {seed}")

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_dir():
        pt_files = sorted(input_path.glob("**/*.pt"))
    else:
        pt_files = [input_path]

    # Limit number of samples for quick testing
    if args.sample is not None:
        pt_files = sorted(random.sample(pt_files, min(args.sample, len(pt_files))))
        logger.info(f"--sample {args.sample}: randomly selected {len(pt_files)} file(s)")

    is_dir_input = input_path.is_dir()

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmflow_repo: str | None = None
    gmflow_ckpt: str | None = None
    if online_opw:
        try:
            resolved_repo, resolved_ckpt = resolve_gmflow_paths()
        except (ImportError, FileNotFoundError) as exc:
            parser.error(str(exc))
        gmflow_repo = str(resolved_repo)
        gmflow_ckpt = str(resolved_ckpt)
        logger.info(
            "Online OPW enabled with auto-detected GMFlow: "
            f"repo={gmflow_repo}, checkpoint={gmflow_ckpt}"
        )
    else:
        logger.info("OPW not requested; summary.json will omit avg_metrics.opw.")

    # Auto-detect GPUs from CUDA_VISIBLE_DEVICES / torch.cuda.device_count()
    n_visible = torch.cuda.device_count()
    gpu_ids = list(range(n_visible)) if n_visible > 0 else [0]

    num_gpus = len(gpu_ids)
    logger.info(
        f"Config: {config['model_name']}+{config['tuning_mode']}, "
        f"{len(pt_files)} file(s), {num_gpus} GPU(s) {gpu_ids}"
    )

    # ---- Single GPU path (original behavior, no subprocess overhead) ----
    if num_gpus <= 1:
        device = torch.device(f"cuda:{gpu_ids[0]}")

        all_metrics, all_times = process_samples(
            rank=0,
            pt_files=pt_files,
            config=config,
            output_dir=output_dir,
            device=device,
            save_vis=args.save_vis,
            save_pt=save_pt,
            fps=args.fps,
            online_opw=online_opw,
            opw_beta=50.0,
            opw_fb_consistency=args.opw_fb_consistency,
            opw_flow_batch_size=args.opw_flow_batch_size,
            opw_flow_max_side=args.opw_flow_max_side,
            save_trace=args.save_trace,
        )
    # ---- Multi-GPU path ----
    else:
        tmp_dir = output_dir / ".tmp_multigpu"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Use fork-server to avoid CUDA re-init issues
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            _worker_fn,
            args=(
                gpu_ids,
                pt_files,
                config,
                output_dir,
                args.save_vis,
                save_pt,
                args.fps,
                tmp_dir,
                online_opw,
                50.0,
                args.opw_fb_consistency,
                args.opw_flow_batch_size,
                args.opw_flow_max_side,
                args.save_trace,
            ),
            nprocs=num_gpus,
            join=True,
        )

        # Aggregate results from all workers
        all_metrics: list[dict] = []
        all_times: list[float] = []
        for rank in range(num_gpus):
            result_file = tmp_dir / f"results_rank{rank}.pt"
            if result_file.exists():
                partial = torch.load(result_file, weights_only=False, map_location="cpu")
                all_metrics.extend(partial["metrics"])
                all_times.extend(partial["times"])
                result_file.unlink()
        # Clean up temp dir
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        # Sort metrics by sample name for deterministic output
        all_metrics.sort(key=lambda m: m.get("sample", ""))

    # Write summary JSON for both directory and single-file inputs with metrics.
    if all_metrics:
        avg = average_metrics(all_metrics, ignore_keys=["sample", "time_s"])
        summary = {
            "config": args.config,
            "model_name": config["model_name"],
            "tuning_mode": config["tuning_mode"],
            "n_steps": config.get("n_steps"),
            "streaming": config.get("streaming"),
            "input": str(input_path),
            "input_dir": str(input_path) if is_dir_input else None,
            "is_dir_input": is_dir_input,
            "num_samples": len(all_metrics),
            "num_gpus": num_gpus,
            "gpu_ids": gpu_ids,
            "avg_metrics": {k: round(v, 6) for k, v in avg.items()},
            "opw_evaluation": _build_opw_evaluation_metadata(
                online_opw=online_opw,
                avg_metrics=avg,
                per_sample=all_metrics,
                gmflow_ckpt=gmflow_ckpt,
                gmflow_repo=gmflow_repo,
                beta=50.0,
                fb_consistency=args.opw_fb_consistency,
                flow_batch_size=args.opw_flow_batch_size,
                flow_max_side=args.opw_flow_max_side,
            ),
        }
        if all_times:
            summary["timing"] = {
                "wall_total_s": round(sum(all_times) / num_gpus, 3),  # approx wall time
                "sum_s": round(sum(all_times), 3),
                "avg_s": round(sum(all_times) / len(all_times), 3),
                "min_s": round(min(all_times), 3),
                "max_s": round(max(all_times), 3),
                "num_timed": len(all_times),
            }
        vram_values = [m["peak_vram_mb"] for m in all_metrics if "peak_vram_mb" in m]
        if vram_values:
            summary["vram"] = {
                "avg_peak_mb": round(sum(vram_values) / len(vram_values), 1),
                "max_peak_mb": round(max(vram_values), 1),
                "min_peak_mb": round(min(vram_values), 1),
            }
        summary["per_sample"] = all_metrics

        summary_path = output_dir / "summary.json"
        temporary = summary_path.with_name(f".{summary_path.name}.tmp")
        temporary.write_text(
            json.dumps(
                _json_safe(summary),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)
        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_path}")
        print(f"Samples: {len(all_metrics)}  |  GPUs: {num_gpus}  |  Avg metrics:")
        for k, v in avg.items():
            print(f"  {k:>10s}: {v:.4f}")
        if all_times:
            print(f"  {'avg_time':>10s}: {sum(all_times)/len(all_times):.2f}s")
            print(f"  {'wall_est':>10s}: ~{sum(all_times)/num_gpus:.2f}s")
        if vram_values:
            print(f"  {'avg_vram':>10s}: {sum(vram_values)/len(vram_values):.0f} MB")
            print(f"  {'max_vram':>10s}: {max(vram_values):.0f} MB")
        print(f"{'='*60}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
