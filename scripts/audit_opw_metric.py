#!/usr/bin/env python
"""Compute OPW from cached predictions and optionally update summary.json.

Recommended Metropolis invocation from the repository root::

    export GMFLOW_REPO=/home/tankh/gmflow
    export GMFLOW_CKPT=$GMFLOW_REPO/pretrained/models/gmflow_sintel-0c07dcb3.pth
    export PYTHONPATH="$GMFLOW_REPO:$PYTHONPATH"

    CUDA_VISIBLE_DEVICES=7 python -u scripts/audit_opw_metric.py \
      --gmflow-ckpt "$GMFLOW_CKPT" \
      --gmflow-repo "$GMFLOW_REPO" \
      --input-dir dataset/metropolis/metropolis_8line_noisy_v3 \
      --pred-dir output/noise_probe/metropolis_8line_v3/vggt \
      --flow-batch-size 2 \
      --fb-consistency \
      --out-json output/noise_probe/metropolis_8line_v3/vggt/opw_capa_strict_fb.json \
      --out-tsv output/noise_probe/metropolis_8line_v3/vggt/opw_capa_strict_fb.tsv \
      --update-summary output/noise_probe/metropolis_8line_v3/vggt/summary.json

Do not type OPW values into summary.json manually. ``--update-summary`` checks
scene coverage and protocol metadata, backs up the old summary, and writes the
audited values atomically.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capa.utils.metric import compute_opw, load_gmflow
from run import load_sample


def _is_finite_number(value: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _json_number(value: float) -> float | None:
    return float(value) if _is_finite_number(value) else None


def _format_opw(value: float) -> str:
    return f"{value:.6f}" if value == value else "nan"


def _collect_pairs(input_dir: Path, pred_dir: Path) -> list[tuple[str, Path, Path]]:
    sample_files = sorted(input_dir.glob("*.pt"))
    if not sample_files:
        raise FileNotFoundError(f"No .pt input samples found in input-dir: {input_dir}")
    pred_files = sorted(pred_dir.glob("*_pred.pt"))
    if not pred_files:
        raise FileNotFoundError(
            f"No *_pred.pt files found in pred-dir: {pred_dir}. "
            "Expected files matching '*_pred.pt'."
        )

    sample_by_stem = {path.stem: path for path in sample_files}
    pred_by_stem = {
        path.stem[: -len("_pred")]: path
        for path in pred_files
    }
    missing_predictions = sorted(set(sample_by_stem) - set(pred_by_stem))
    extra_predictions = sorted(set(pred_by_stem) - set(sample_by_stem))
    if missing_predictions or extra_predictions:
        raise FileNotFoundError(
            "Input and prediction scene coverage differ: "
            f"missing predictions={missing_predictions}, "
            f"extra predictions={extra_predictions}"
        )
    return [
        (stem, sample_by_stem[stem], pred_by_stem[stem])
        for stem in sorted(sample_by_stem)
    ]


def _collect_pairs_from_input_slice(
    input_dir: Path,
    pred_dir: Path,
    scene_offset: int,
    max_scenes: int | None,
) -> list[tuple[str, Path, Path]]:
    sample_files = sorted(input_dir.glob("*.pt"))
    if not sample_files:
        raise FileNotFoundError(f"No .pt input samples found in input-dir: {input_dir}")

    sample_files = sample_files[scene_offset:]
    if max_scenes is not None:
        sample_files = sample_files[:max_scenes]

    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for sample_path in sample_files:
        stem = sample_path.stem
        pred_path = pred_dir / f"{stem}_pred.pt"
        if not pred_path.exists():
            missing.append(stem)
            continue
        pairs.append((stem, sample_path, pred_path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} prediction(s) under pred-dir: {pred_dir}. "
            f"Example missing prediction: {missing[0]}_pred.pt"
        )
    return pairs


def _load_depth(pred_path: Path, depth_key: str, device: torch.device, stem: str) -> torch.Tensor:
    saved = torch.load(pred_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        raise KeyError(
            f"{stem}: prediction file {pred_path} must contain a dict with key "
            f"'{depth_key}', got {type(saved).__name__}"
        )
    if depth_key not in saved:
        available = ", ".join(sorted(map(str, saved.keys())))
        raise KeyError(
            f"{stem}: missing depth key '{depth_key}' in prediction file {pred_path}. "
            f"Available keys/type: {available}"
        )
    return saved[depth_key].float().to(device)


def _load_sample_tensors(
    sample_path: Path,
    device: torch.device,
    stem: str,
    eval_mask_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        data = load_sample(str(sample_path))
    except KeyError as exc:
        raise KeyError(
            f"{stem}: load_sample({sample_path}) failed to provide required key "
            "'rgb_nv3hw'. If the sample uses source_pt/source_base compact storage, "
            "load_sample() must decode it before OPW evaluation."
        ) from exc

    if "rgb_nv3hw" not in data:
        available = ", ".join(sorted(map(str, data.keys())))
        raise KeyError(
            f"{stem}: missing RGB key 'rgb_nv3hw' in input sample {sample_path}. "
            f"Available keys: {available}"
        )
    if eval_mask_key not in data:
        available = ", ".join(sorted(map(str, data.keys())))
        raise KeyError(
            f"{stem}: missing dense ground-truth key '{eval_mask_key}' in input "
            f"sample {sample_path}. CAPA strict OPW defines Omega from valid "
            f"dense ground-truth pixels. Available keys: {available}"
        )

    rgb = data["rgb_nv3hw"].float().to(device)
    depth_gt = data[eval_mask_key].float().to(device)
    eval_mask = torch.isfinite(depth_gt) & (depth_gt > 0)
    return rgb, eval_mask


def _check_shapes(
    stem: str,
    depth_pred: torch.Tensor,
    rgb: torch.Tensor,
    eval_mask: torch.Tensor,
    depth_key: str,
    eval_mask_key: str,
) -> None:
    if depth_pred.ndim != 3:
        raise ValueError(
            f"{stem}: saved['{depth_key}'] must have shape [T,H,W], "
            f"got {tuple(depth_pred.shape)}"
        )
    if rgb.ndim != 4:
        raise ValueError(
            f"{stem}: data['rgb_nv3hw'] must have shape [T,3,H,W], "
            f"got {tuple(rgb.shape)}"
        )
    if rgb.shape[1] != 3:
        raise ValueError(
            f"{stem}: data['rgb_nv3hw'] must have RGB channel dimension 3 at axis 1, "
            f"got {tuple(rgb.shape)}"
        )
    if depth_pred.shape[0] != rgb.shape[0]:
        raise ValueError(
            f"{stem}: temporal shape mismatch: depth T={depth_pred.shape[0]}, rgb T={rgb.shape[0]}"
        )
    if depth_pred.shape[-2:] != rgb.shape[-2:]:
        raise ValueError(
            f"{stem}: spatial shape mismatch: depth H,W={tuple(depth_pred.shape[-2:])}, "
            f"rgb H,W={tuple(rgb.shape[-2:])}"
        )
    if eval_mask.shape != depth_pred.shape:
        raise ValueError(
            f"{stem}: data['{eval_mask_key}'] must match saved['{depth_key}'] "
            f"shape {tuple(depth_pred.shape)}, got {tuple(eval_mask.shape)}"
        )


def _print_pair_details(stem: str, details: dict[str, Any]) -> None:
    per_pair = details.get("per_pair_opw", [])
    valid_count = details.get("valid_count", [])
    weight_sum = details.get("weight_sum", [])
    flow_valid_count = details.get("flow_valid_count", [])
    valid_weight_count = details.get("valid_weight_count", [])
    invalid_correspondence_count = details.get("invalid_correspondence_count", [])
    invalid_warped_depth_count = details.get("invalid_warped_depth_count", [])

    for idx, pair_opw in enumerate(per_pair):
        vc = valid_count[idx].item() if idx < len(valid_count) else float("nan")
        ws = weight_sum[idx].item() if idx < len(weight_sum) else float("nan")
        fvc = flow_valid_count[idx].item() if idx < len(flow_valid_count) else float("nan")
        vwc = (
            valid_weight_count[idx].item()
            if idx < len(valid_weight_count)
            else float("nan")
        )
        icc = (
            invalid_correspondence_count[idx].item()
            if idx < len(invalid_correspondence_count)
            else float("nan")
        )
        iwd = (
            invalid_warped_depth_count[idx].item()
            if idx < len(invalid_warped_depth_count)
            else float("nan")
        )
        opw_value = pair_opw.item() if hasattr(pair_opw, "item") else float(pair_opw)
        print(
            f"{stem}\tpair={idx}\topw={_format_opw(opw_value)}"
            f"\tvalid_count={vc}\tflow_valid_count={fvc}"
            f"\tvalid_weight_count={vwc}\tinvalid_correspondence_count={icc}"
            f"\tinvalid_warped_depth_count={iwd}\tweight_sum={ws:.6f}"
        )


def _write_tsv(path: Path, scene_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        f.write("scene\topw\n")
        for item in scene_results:
            f.write(f"{item['scene']}\t{_format_opw(item['opw'])}\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-compute OPW from cached predictions via compute_opw()."
    )
    parser.add_argument(
        "--gmflow-ckpt",
        default=None,
        help=(
            "Path to GMFlow checkpoint (.pth). If omitted, load_gmflow() tries "
            "GMFLOW_CKPT and the GMFlow pretrained/models directory."
        ),
    )
    parser.add_argument(
        "--gmflow-repo",
        type=str,
        default=None,
        help="Optional external GMFlow repository path passed to load_gmflow()",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Directory containing input sample .pt files",
    )
    parser.add_argument(
        "--pred-dir",
        "-p",
        type=str,
        required=True,
        help="Directory containing cached *_pred.pt files",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail immediately if CUDA is unavailable",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=50.0,
        help="RGB visibility weight beta passed to compute_opw()",
    )
    parser.add_argument(
        "--flow-batch-size",
        type=int,
        default=None,
        help=(
            "Maximum adjacent frame pairs per GMFlow call. Use 2 or 4 for "
            "high-resolution Metropolis clips."
        ),
    )
    parser.add_argument(
        "--flow-max-side",
        type=int,
        default=None,
        help=(
            "Optional maximum image side for GMFlow inference. Flow is resized "
            "and vector-scaled back to the original evaluation grid."
        ),
    )
    parser.add_argument(
        "--fb-consistency",
        action="store_true",
        help=(
            "Enable optional forward-backward flow consistency filtering. "
            "Disabled by default for formula-strict CAPA OPW."
        ),
    )
    parser.add_argument(
        "--no-fb-consistency",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--opw-mode",
        default="capa_strict",
        help="OPW mode passed to compute_opw() (default: capa_strict)",
    )
    parser.add_argument(
        "--depth-key",
        default="depth_pred_nhw",
        help="Prediction tensor key inside each *_pred.pt file (default: depth_pred_nhw)",
    )
    parser.add_argument(
        "--eval-mask-key",
        default="depth_gt_nvhw",
        help=(
            "Dense ground-truth depth key used to define Omega as finite values > 0 "
            "(default: depth_gt_nvhw)"
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Only process the first N matched scenes for local/server debugging",
    )
    parser.add_argument(
        "--scene-offset",
        type=int,
        default=0,
        help="Skip the first N matched scenes after sorting, before --max-scenes",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--update-summary",
        type=Path,
        default=None,
        help=(
            "After a successful audit, merge --out-json into this summary.json. "
            "Exact scene coverage is required and the summary is backed up."
        ),
    )
    parser.add_argument("--out-tsv", type=str, default=None, help="Optional TSV output path")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-pair OPW diagnostics when compute_opw(return_details=True) is available",
    )
    args = parser.parse_args()

    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError(f"--max-scenes must be positive, got {args.max_scenes}")
    if args.scene_offset < 0:
        raise ValueError(f"--scene-offset must be non-negative, got {args.scene_offset}")
    if args.flow_batch_size is not None and args.flow_batch_size <= 0:
        raise ValueError(
            f"--flow-batch-size must be positive, got {args.flow_batch_size}"
        )
    if args.flow_max_side is not None and args.flow_max_side <= 0:
        raise ValueError(f"--flow-max-side must be positive, got {args.flow_max_side}")
    if not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError(f"--beta must be a finite positive number, got {args.beta!r}")
    if args.update_summary is not None and args.out_json is None:
        raise ValueError("--update-summary requires --out-json")

    if args.opw_mode != "capa_strict":
        raise NotImplementedError(
            f"Unsupported --opw-mode {args.opw_mode!r}. Only 'capa_strict' is implemented."
        )

    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require-gpu was set, but CUDA is unavailable.")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: CUDA is unavailable; falling back to CPU. "
            "Use --require-gpu to fail instead.",
            file=sys.stderr,
        )
        device = torch.device("cpu")
    else:
        device = requested_device

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    if args.scene_offset or args.max_scenes is not None:
        pairs = _collect_pairs_from_input_slice(
            input_dir,
            pred_dir,
            args.scene_offset,
            args.max_scenes,
        )
    else:
        pairs = _collect_pairs(input_dir, pred_dir)

    fb_consistency = bool(args.fb_consistency)
    if args.fb_consistency and args.no_fb_consistency:
        raise ValueError("--fb-consistency and deprecated --no-fb-consistency cannot both be set.")
    print(f"fb_consistency: {str(fb_consistency).lower()}")

    try:
        flow_model = load_gmflow(
            args.gmflow_ckpt,
            device,
            gmflow_repo=args.gmflow_repo,
        )
    except ImportError as exc:
        print(f"ERROR: GMFlow unavailable: {exc}", file=sys.stderr)
        return 2

    resolved_gmflow_repo = getattr(flow_model, "_capa_gmflow_repo", args.gmflow_repo)
    resolved_gmflow_ckpt = getattr(flow_model, "_capa_gmflow_ckpt", args.gmflow_ckpt)
    print(f"gmflow_repo: {resolved_gmflow_repo}")
    print(f"gmflow_ckpt: {resolved_gmflow_ckpt}")

    opw_values: list[float] = []
    scene_results: list[dict[str, Any]] = []
    for stem, sample_path, pred_path in pairs:
        depth_pred = _load_depth(pred_path, args.depth_key, device, stem)
        rgb, eval_mask = _load_sample_tensors(
            sample_path,
            device,
            stem,
            args.eval_mask_key,
        )
        _check_shapes(
            stem,
            depth_pred,
            rgb,
            eval_mask,
            args.depth_key,
            args.eval_mask_key,
        )

        if args.verbose:
            opw_result = compute_opw(
                depth_pred,
                rgb,
                flow_model=flow_model,
                beta=args.beta,
                fb_consistency=fb_consistency,
                return_details=True,
                opw_mode=args.opw_mode,
                eval_mask=eval_mask,
                flow_batch_size=args.flow_batch_size,
                flow_max_side=args.flow_max_side,
            )
            opw, details = opw_result
            _print_pair_details(stem, details)
        else:
            opw = compute_opw(
                depth_pred,
                rgb,
                flow_model=flow_model,
                beta=args.beta,
                fb_consistency=fb_consistency,
                opw_mode=args.opw_mode,
                eval_mask=eval_mask,
                flow_batch_size=args.flow_batch_size,
                flow_max_side=args.flow_max_side,
            )

        if not _is_finite_number(opw):
            raise RuntimeError(f"Non-finite OPW for scene {stem}: {opw!r}")
        opw = float(opw)
        opw_values.append(opw)
        scene_results.append({"scene": stem, "opw": opw})
        print(f"{stem}\t{_format_opw(opw)}")

    if not opw_values:
        raise RuntimeError("No OPW values were computed")
    mean_opw = sum(opw_values) / len(opw_values)
    print(f"mean\t{mean_opw:.6f}\t({len(opw_values)}/{len(opw_values)} scenes)")

    if args.out_tsv:
        _write_tsv(Path(args.out_tsv), scene_results)

    if args.out_json:
        payload = {
            "input_dir": str(input_dir),
            "pred_dir": str(pred_dir),
            "gmflow_ckpt": (
                str(resolved_gmflow_ckpt) if resolved_gmflow_ckpt is not None else None
            ),
            "gmflow_repo": str(resolved_gmflow_repo) if resolved_gmflow_repo is not None else None,
            "beta": float(args.beta),
            "fb_consistency": bool(fb_consistency),
            "opw_mode": args.opw_mode,
            "depth_key": args.depth_key,
            "eval_mask_key": args.eval_mask_key,
            "flow_batch_size": args.flow_batch_size,
            "flow_max_side": args.flow_max_side,
            "per_scene_opw": [
                {"scene": item["scene"], "opw": _json_number(item["opw"])}
                for item in scene_results
            ],
            "mean_opw": _json_number(mean_opw),
            "num_valid_scenes": len(opw_values),
            "num_total_scenes": len(opw_values),
        }
        _write_json(Path(args.out_json), payload)
        if args.update_summary is not None:
            from scripts.merge_opw_summary import merge_summary

            merge_summary(args.update_summary, Path(args.out_json), None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
