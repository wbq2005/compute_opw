#!/usr/bin/env python
"""Compute OPW from cached *_pred.pt files using capa.utils.metric.compute_opw."""
from __future__ import annotations
r"""
cd /home/tankh/capa_reproduce

CUDA_VISIBLE_DEVICES=0 python scripts/audit_opw_metric.py \
  --gmflow-ckpt /home/tankh/gmflow/pretrained/models/gmflow_sintel-0c07dcb3.pth \
  --input-dir dataset/scannet/scannet_sift_noisy \ 
  --pred-dir output/tab1_zeroshot_vggt_baselines/scannet_sift/vggt

CUDA_VISIBLE_DEVICES=1 python scripts/audit_opw_metric.py \
  --gmflow-ckpt /home/tankh/gmflow/pretrained/models/gmflow_sintel-0c07dcb3.pth \
  --input-dir /home/tankh/capa_reproduce/dataset/metropolis/metropolis_8line_noisy \
  --pred-dir output/tab1_zeroshot_vggt_baselines/metropolis_8line/vggt
"""

import argparse
import json
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
    return value == value and value not in (float("inf"), float("-inf"))


def _json_number(value: float) -> float | None:
    return float(value) if _is_finite_number(value) else None


def _format_opw(value: float) -> str:
    return f"{value:.6f}" if value == value else "nan"


def _collect_pairs(input_dir: Path, pred_dir: Path) -> list[tuple[str, Path, Path]]:
    pred_files = sorted(pred_dir.glob("*_pred.pt"))
    if not pred_files:
        raise FileNotFoundError(
            f"No *_pred.pt files found in pred-dir: {pred_dir}. "
            "Expected files matching '*_pred.pt'."
        )

    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for pred_path in pred_files:
        stem = pred_path.stem[: -len("_pred")]
        sample_path = input_dir / f"{stem}.pt"
        if not sample_path.exists():
            missing.append(stem)
            continue
        pairs.append((stem, sample_path, pred_path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} input sample(s) under input-dir: {input_dir}. "
            f"Example missing sample: {missing[0]}.pt"
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


def _load_rgb(sample_path: Path, device: torch.device, stem: str) -> torch.Tensor:
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
    return data["rgb_nv3hw"].float().to(device)


def _check_shapes(stem: str, depth_pred: torch.Tensor, rgb: torch.Tensor, depth_key: str) -> None:
    if depth_pred.ndim != 3:
        raise ValueError(
            f"{stem}: saved['{depth_key}'] must have shape [T,H,W], "
            f"got {tuple(depth_pred.shape)}"
        )
    if rgb.ndim != 4:
        raise ValueError(f"{stem}: data['rgb_nv3hw'] must have shape [T,3,H,W], got {tuple(rgb.shape)}")
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


def _print_pair_details(stem: str, details: dict[str, Any]) -> None:
    per_pair = details.get("per_pair_opw", [])
    valid_count = details.get("valid_count", [])
    weight_sum = details.get("weight_sum", [])

    for idx, pair_opw in enumerate(per_pair):
        vc = valid_count[idx].item() if idx < len(valid_count) else float("nan")
        ws = weight_sum[idx].item() if idx < len(weight_sum) else float("nan")
        opw_value = pair_opw.item() if hasattr(pair_opw, "item") else float(pair_opw)
        print(
            f"{stem}\tpair={idx}\topw={_format_opw(opw_value)}"
            f"\tvalid_count={vc}\tweight_sum={ws:.6f}"
        )


def _write_tsv(path: Path, scene_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("scene\topw\n")
        for item in scene_results:
            f.write(f"{item['scene']}\t{_format_opw(item['opw'])}\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-compute OPW from cached predictions via compute_opw()."
    )
    parser.add_argument("--gmflow-ckpt", required=True, help="Path to GMFlow checkpoint (.pth)")
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
        "--max-scenes",
        type=int,
        default=None,
        help="Only process the first N matched scenes for local/server debugging",
    )
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON summary output path")
    parser.add_argument("--out-tsv", type=str, default=None, help="Optional TSV output path")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-pair OPW diagnostics when compute_opw(return_details=True) is available",
    )
    args = parser.parse_args()

    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError(f"--max-scenes must be positive, got {args.max_scenes}")

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
    pairs = _collect_pairs(input_dir, pred_dir)
    if args.max_scenes is not None:
        pairs = pairs[: args.max_scenes]

    fb_consistency = bool(args.fb_consistency)
    if args.fb_consistency and args.no_fb_consistency:
        raise ValueError("--fb-consistency and deprecated --no-fb-consistency cannot both be set.")
    print(f"fb_consistency: {str(fb_consistency).lower()}")

    try:
        flow_model = load_gmflow(args.gmflow_ckpt, device, gmflow_repo=args.gmflow_repo)
    except ImportError as exc:
        print(f"ERROR: GMFlow unavailable: {exc}", file=sys.stderr)
        return 2

    opw_values: list[float] = []
    scene_results: list[dict[str, Any]] = []
    for stem, sample_path, pred_path in pairs:
        depth_pred = _load_depth(pred_path, args.depth_key, device, stem)
        rgb = _load_rgb(sample_path, device, stem)
        _check_shapes(stem, depth_pred, rgb, args.depth_key)

        if args.verbose:
            opw_result = compute_opw(
                depth_pred,
                rgb,
                flow_model=flow_model,
                beta=args.beta,
                fb_consistency=fb_consistency,
                return_details=True,
                opw_mode=args.opw_mode,
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
            )

        opw_values.append(opw)
        scene_results.append({"scene": stem, "opw": opw})
        print(f"{stem}\t{_format_opw(opw)}")

    valid = [v for v in opw_values if _is_finite_number(v)]
    mean_opw = float("nan")
    if valid:
        mean_opw = sum(valid) / len(valid)
        print(f"mean\t{mean_opw:.6f}\t({len(valid)}/{len(opw_values)} scenes)")

    if args.out_tsv:
        _write_tsv(Path(args.out_tsv), scene_results)

    if args.out_json:
        payload = {
            "input_dir": str(input_dir),
            "pred_dir": str(pred_dir),
            "gmflow_ckpt": str(args.gmflow_ckpt),
            "gmflow_repo": str(args.gmflow_repo) if args.gmflow_repo is not None else None,
            "beta": float(args.beta),
            "fb_consistency": bool(fb_consistency),
            "opw_mode": args.opw_mode,
            "depth_key": args.depth_key,
            "per_scene_opw": [
                {"scene": item["scene"], "opw": _json_number(item["opw"])}
                for item in scene_results
            ],
            "mean_opw": _json_number(mean_opw),
            "num_valid_scenes": len(valid),
            "num_total_scenes": len(opw_values),
        }
        _write_json(Path(args.out_json), payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
