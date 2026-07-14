# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""CAPA metric implementations.

Canonical offline OPW evaluation command, run from the repository root::

    CUDA_VISIBLE_DEVICES=<gpu_id> python -u scripts/audit_opw_metric.py \
      --input-dir <dataset_dir> \
      --pred-dir <prediction_dir> \
      --flow-batch-size <2_or_4>

Use flow batch size 4 for ScanNet/7-Scenes and 2 for high-resolution
Metropolis clips. GMFlow and its checkpoint are discovered automatically. A
forward/backward consistency check is enabled by default; use
``--no-fb-consistency`` only for an explicitly labelled ablation. A
complete audit writes ``opw.json``/``opw.tsv``, backs up the adjacent
``summary.json``, and merges the audited OPW values into it. Commands using
``--max-scenes`` or ``--scene-offset`` are probes and never auto-update the
formal summary.
"""

import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .logging import get_local_logger

logger = get_local_logger(__name__)

OPW_PROTOCOL = "capa_strict"
DEFAULT_FB_CONSISTENCY = True

_GMFLOW_HELP = (
    "GMFlow is required for real OPW evaluation but could not be imported. "
    "Clone https://github.com/haofeixu/gmflow, set GMFLOW_REPO=/path/to/gmflow, "
    "or place the checkout at ~/gmflow or third_party/gmflow. Set GMFLOW_CKPT "
    "only for a non-standard checkpoint location; otherwise keep "
    "gmflow_sintel-0c07dcb3.pth under GMFlow pretrained/models."
)


def mask_aware_batch_mean(
    tensor: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Mean over spatial dims with per-sample valid-pixel masking.

    Args:
        tensor: [B, H, W] value map.
        valid_mask: [B, H, W] boolean mask (True = valid).

    Returns:
        Scalar tensor (mean across all batches).
    """
    B, H, W = tensor.shape
    tensor = tensor.clone()

    if valid_mask is not None:
        tensor[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))  # [B]

        if (n == 0).any():
            logger.warning("Some batches have no valid pixels. Counts: %s", n.tolist())
            n_safe = torch.clamp(n, min=1)
            result = torch.sum(tensor, (-1, -2)) / n_safe
            result = torch.where(
                n > 0,
                result,
                torch.tensor(float("nan"), dtype=result.dtype, device=result.device),
            )
            return torch.nanmean(result)
        return torch.sum(tensor, (-1, -2)) / n
    else:
        return torch.sum(tensor, (-1, -2)) / (H * W)


def _resolve_gmflow_repo(gmflow_repo: str | os.PathLike[str] | None = None) -> Path | None:
    """Resolve an external GMFlow repository path without vendoring it into ``capa``."""
    candidates: list[Path] = []

    if gmflow_repo is not None:
        candidates.append(Path(gmflow_repo).expanduser())

    env_repo = os.environ.get("GMFLOW_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            Path.home() / "gmflow",
            repo_root / "third_party" / "gmflow",
            repo_root / "gmflow",
            repo_root / "gmflow(1)" / "gmflow",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "gmflow" / "gmflow.py").exists():
            return candidate
        if candidate.name == "gmflow" and (candidate / "gmflow.py").exists():
            return candidate.parent

    return None


def _resolve_gmflow_ckpt(
    ckpt_path: str | os.PathLike[str] | None,
    gmflow_repo: Path | None,
) -> Path | None:
    """Resolve the GMFlow Sintel checkpoint from an explicit path, env, or repo."""
    candidates: list[Path] = []

    if ckpt_path is not None:
        candidates.append(Path(ckpt_path).expanduser())

    env_ckpt = os.environ.get("GMFLOW_CKPT")
    if env_ckpt:
        candidates.append(Path(env_ckpt).expanduser())

    if gmflow_repo is not None:
        candidates.extend(
            [
                gmflow_repo / "pretrained" / "models" / "gmflow_sintel-0c07dcb3.pth",
                gmflow_repo / "pretrained" / "gmflow_sintel-0c07dcb3.pth",
            ]
        )
        candidates.extend(
            sorted((gmflow_repo / "pretrained" / "models").glob("gmflow_sintel-*.pth"))
        )
        candidates.extend(sorted((gmflow_repo / "pretrained").glob("gmflow_sintel-*.pth")))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            Path.home()
            / "gmflow"
            / "pretrained"
            / "models"
            / "gmflow_sintel-0c07dcb3.pth",
            repo_root
            / "gmflow(1)"
            / "gmflow"
            / "pretrained"
            / "models"
            / "gmflow_sintel-0c07dcb3.pth",
            repo_root
            / "third_party"
            / "gmflow"
            / "pretrained"
            / "models"
            / "gmflow_sintel-0c07dcb3.pth",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    return None


def resolve_gmflow_paths(
    ckpt_path: str | os.PathLike[str] | None = None,
    gmflow_repo: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Resolve the GMFlow checkout and Sintel checkpoint without loading CUDA."""
    resolved_repo = _resolve_gmflow_repo(gmflow_repo)
    if resolved_repo is None:
        raise ImportError(_GMFLOW_HELP)

    resolved_ckpt = _resolve_gmflow_ckpt(ckpt_path, resolved_repo)
    if resolved_ckpt is None:
        requested = Path(ckpt_path).expanduser() if ckpt_path is not None else None
        raise FileNotFoundError(
            f"GMFlow checkpoint not found: {requested}. "
            "Set GMFLOW_CKPT or keep gmflow_sintel-0c07dcb3.pth under "
            "the detected GMFlow pretrained/models directory."
        )
    return resolved_repo, resolved_ckpt


def load_gmflow(
    ckpt_path: str | os.PathLike[str] | None = None,
    device: torch.device | str = "cuda",
    gmflow_repo: str | os.PathLike[str] | None = None,
) -> torch.nn.Module:
    """
    Load GMFlow from an external checkout for real OPW evaluation.

    The GMFlow source is deliberately not vendored inside ``capa``.  Resolution
    order is:
      1. explicit ``gmflow_repo`` argument;
      2. ``GMFLOW_REPO`` environment variable;
      3. ``~/gmflow``;
      4. ``third_party/gmflow`` or ``gmflow`` under the project root.

    Raises:
        ImportError: if GMFlow cannot be imported.
        FileNotFoundError: if the checkpoint path does not exist.
    """
    resolved_repo, ckpt = resolve_gmflow_paths(ckpt_path, gmflow_repo)
    if str(resolved_repo) not in sys.path:
        sys.path.insert(0, str(resolved_repo))

    try:
        from gmflow.gmflow import GMFlow
    except Exception as exc:  # pragma: no cover - depends on external GMFlow
        raise ImportError(_GMFLOW_HELP) from exc

    device = torch.device(device)
    model = GMFlow(
        num_scales=1,
        upsample_factor=8,
        feature_channels=128,
        attention_type="swin",
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
    )

    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    state_dict: Any
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported GMFlow checkpoint format: {ckpt}")

    cleaned_state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Official GMFlow normalizes images internally by dividing by 255.  The OPW
    # visibility computation uses RGB in [0, 1], so compute_opw scales images
    # back to [0, 255] only for models loaded through this helper.
    model._capa_expects_rgb_255 = True  # type: ignore[attr-defined]
    model._capa_pad_to_multiple = 16  # type: ignore[attr-defined]
    model._capa_gmflow_kwargs = {  # type: ignore[attr-defined]
        "attn_splits_list": [2],
        "corr_radius_list": [-1],
        "prop_radius_list": [-1],
    }
    model._capa_gmflow_repo = (  # type: ignore[attr-defined]
        str(resolved_repo) if resolved_repo is not None else None
    )
    model._capa_gmflow_ckpt = str(ckpt)  # type: ignore[attr-defined]
    return model


def bilinear_warp_by_backward_flow(
    x: torch.Tensor,
    flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinearly warp ``x`` with a backward optical-flow field.

    Args:
        x: source tensor with shape ``[B, C, H, W]``.
        flow: backward flow with shape ``[B, 2, H, W]`` in pixel units.  For
            each target pixel ``(x_t, y_t)``, the source sampling location is
            ``(x_t + flow_x, y_t + flow_y)``.

    Returns:
        ``(warped_x, valid_mask)`` where ``warped_x`` has shape ``[B, C, H, W]``
        and ``valid_mask`` has shape ``[B, H, W]``.  ``valid_mask`` is false
        for non-finite flow values and out-of-bound sampling locations.

    Notes:
        ``align_corners=True`` is fixed intentionally: pixel coordinate ``0``
        maps to normalized coordinate ``-1`` and pixel coordinate ``W-1/H-1``
        maps to ``1``.  This makes the flow values exact pixel displacements
        on the usual image-coordinate grid.
    """
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B,C,H,W], got {tuple(x.shape)}")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must have shape [B,2,H,W], got {tuple(flow.shape)}")
    if x.shape[0] != flow.shape[0] or x.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            "x and flow batch/spatial shapes must match, got "
            f"{tuple(x.shape)} and {tuple(flow.shape)}"
        )

    B, _, H, W = x.shape
    device = x.device
    dtype = x.dtype if x.is_floating_point() else torch.float32
    x_for_sample = x if x.is_floating_point() else x.to(dtype)
    flow = flow.to(device=device, dtype=dtype)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    sample_x = xs.unsqueeze(0) + flow[:, 0]
    sample_y = ys.unsqueeze(0) + flow[:, 1]

    finite = torch.isfinite(sample_x) & torch.isfinite(sample_y)
    valid = finite & (sample_x >= 0) & (sample_x <= W - 1) & (sample_y >= 0) & (sample_y <= H - 1)

    if W > 1:
        grid_x = 2.0 * sample_x / (W - 1) - 1.0
    else:
        grid_x = torch.zeros((B, H, W), device=device, dtype=dtype)
    if H > 1:
        grid_y = 2.0 * sample_y / (H - 1) - 1.0
    else:
        grid_y = torch.zeros((B, H, W), device=device, dtype=dtype)

    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))

    warped = F.grid_sample(
        x_for_sample,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, valid


def _extract_flow_tensor(flow_output: Any) -> torch.Tensor:
    """Extract a ``[B, 2, H, W]`` flow tensor from common flow-model outputs."""
    if isinstance(flow_output, torch.Tensor):
        flow = flow_output
    elif isinstance(flow_output, dict):
        for key in ("flow_preds", "flows", "flow", "flow_pred"):
            if key in flow_output:
                value = flow_output[key]
                flow = value[-1] if isinstance(value, (list, tuple)) else value
                break
        else:
            raise RuntimeError(
                "Flow model output dict must contain one of: flow_preds, flows, flow, flow_pred"
            )
    elif isinstance(flow_output, (list, tuple)) and flow_output:
        flow = flow_output[-1]
    else:
        raise RuntimeError(f"Unsupported flow model output type: {type(flow_output)!r}")

    if not isinstance(flow, torch.Tensor):
        raise RuntimeError(f"Extracted flow is not a tensor: {type(flow)!r}")
    if flow.ndim != 4:
        raise RuntimeError(f"Flow tensor must be 4-D, got {tuple(flow.shape)}")
    if flow.shape[1] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"Flow tensor must have 2 channels, got {tuple(flow.shape)}")


def _pad_to_multiple(
    x: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad symmetrically with replication so H and W are divisible by ``multiple``."""
    if multiple <= 1:
        return x, (0, 0, 0, 0)
    H, W = x.shape[-2:]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(x, pad, mode="replicate"), pad


def _unpad_flow(flow: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """Remove symmetric padding from a flow tensor."""
    pad_left, pad_right, pad_top, pad_bottom = pad
    H, W = flow.shape[-2:]
    end_h = H - pad_bottom if pad_bottom else H
    end_w = W - pad_right if pad_right else W
    return flow[..., pad_top:end_h, pad_left:end_w]


def _run_flow_model(
    flow_model: Any,
    image0: torch.Tensor,
    image1: torch.Tensor,
    batch_size: int | None = None,
    max_side: int | None = None,
) -> torch.Tensor:
    """
    Run a generic flow model from ``image0`` to ``image1`` and return pixel flow.

    Models returned by :func:`load_gmflow` are called with GMFlow's standard
    inference kwargs and images scaled to [0, 255].  Generic unit-test fakes are
    called with the normalized tensors unchanged.
    """
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError(f"flow batch_size must be positive, got {batch_size}")
        if image0.shape[0] > batch_size:
            flows = [
                _run_flow_model(
                    flow_model,
                    image0[start : start + batch_size],
                    image1[start : start + batch_size],
                    max_side=max_side,
                )
                for start in range(0, image0.shape[0], batch_size)
            ]
            return torch.cat(flows, dim=0)

    if max_side is not None and max_side <= 0:
        raise ValueError(f"flow max_side must be positive, got {max_side}")

    original_h, original_w = image0.shape[-2:]
    resized = max_side is not None and max(original_h, original_w) > max_side
    if resized:
        resize_scale = float(max_side) / float(max(original_h, original_w))
        inference_h = max(1, int(round(original_h * resize_scale)))
        inference_w = max(1, int(round(original_w * resize_scale)))
        image0 = F.interpolate(
            image0,
            size=(inference_h, inference_w),
            mode="bilinear",
            align_corners=True,
        )
        image1 = F.interpolate(
            image1,
            size=(inference_h, inference_w),
            mode="bilinear",
            align_corners=True,
        )

    expects_rgb_255 = bool(getattr(flow_model, "_capa_expects_rgb_255", False))
    pad_multiple = int(getattr(flow_model, "_capa_pad_to_multiple", 1))

    image0_for_flow = image0 * 255.0 if expects_rgb_255 else image0
    image1_for_flow = image1 * 255.0 if expects_rgb_255 else image1
    image0_for_flow, pad = _pad_to_multiple(image0_for_flow, pad_multiple)
    if any(pad):
        image1_for_flow = F.pad(image1_for_flow, pad, mode="replicate")

    gmflow_kwargs = getattr(
        flow_model,
        "_capa_gmflow_kwargs",
        {
            "attn_splits_list": [2],
            "corr_radius_list": [-1],
            "prop_radius_list": [-1],
        },
    )
    with torch.no_grad():
        try:
            output = flow_model(image0_for_flow, image1_for_flow, **gmflow_kwargs)
        except TypeError:
            output = flow_model(image0_for_flow, image1_for_flow)

    flow = _extract_flow_tensor(output)
    flow = _unpad_flow(flow, pad)
    if resized:
        inference_h, inference_w = flow.shape[-2:]
        flow = F.interpolate(
            flow,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=True,
        )
        flow[:, 0] *= float(original_w) / float(inference_w)
        flow[:, 1] *= float(original_h) / float(inference_h)
    return flow.to(device=image0.device, dtype=image0.dtype)


def _forward_backward_consistency_mask(
    backward_flow: torch.Tensor,
    forward_flow: torch.Tensor,
) -> torch.Tensor:
    """GMFlow/UnFlow forward-backward consistency mask in target coordinates."""
    warped_forward, valid = bilinear_warp_by_backward_flow(forward_flow, backward_flow)
    fb_error = torch.linalg.vector_norm(backward_flow + warped_forward, dim=1)
    # Match GMFlow's forward_backward_consistency_check exactly: the threshold
    # magnitude uses the two unwarped flow fields at the same array location.
    fb_mag = torch.linalg.vector_norm(backward_flow, dim=1) + torch.linalg.vector_norm(
        forward_flow, dim=1
    )
    return valid & (fb_error <= 0.01 * fb_mag + 0.5)


def compute_opw(
    depth_pred: torch.Tensor,
    rgb: torch.Tensor,
    flow_model: Any | None = None,
    beta: float = 50.0,
    fb_consistency: bool = DEFAULT_FB_CONSISTENCY,
    return_details: bool = False,
    eval_mask: torch.Tensor | None = None,
    flow_batch_size: int | None = None,
    flow_max_side: int | None = None,
) -> float | tuple[float, dict[str, Any]]:
    """
    Compute CAPA's optical-flow-based warping error (OPW).

    Args:
        depth_pred: predicted metric depth with shape ``[T, H, W]``.
        rgb: RGB frames with shape ``[T, 3, H, W]``.  Values should be float
            in ``[0, 1]``; if the maximum finite value is greater than 2, the
            tensor is automatically divided by 255.
        flow_model: optional callable/module that predicts flow from image0 to image1.
            When omitted, GMFlow and its Sintel checkpoint are discovered from
            the standard repository locations and loaded automatically.
            For each adjacent pair, OPW calls it as ``flow_model(I[t+1], I[t])``
            to obtain the backward flow ``F_{t+1=>t}``.
        beta: visibility-weight coefficient.  CAPA uses 50.
        fb_consistency: forward/backward consistency filter. Enabled by default
            for the canonical CAPA evaluation protocol.
        return_details: if true, return ``(opw, details)``.  ``details`` contains
            reported-scale per-pair OPW values (also multiplied by 100),
            denominator valid counts, weight sums, beta, protocol, and whether
            forward/backward consistency was used.
        eval_mask: evaluated-pixel mask ``Omega`` with shape ``[T,H,W]``.
            CAPA defines this as the valid dense ground-truth depth mask. It is
            required in strict mode and is never inferred from the prediction.
        flow_batch_size: optional maximum number of adjacent frame pairs passed
            to the flow model at once. It controls peak memory without changing
            the OPW formula.
        flow_max_side: optional maximum image side for flow inference. The flow
            is resized and vector-scaled back to the original pixel grid using
            GMFlow's official inference-size convention.

    Returns:
        Reported OPW multiplied by 100.  In CAPA strict mode, depth is used in
        metric scale without median/mean/MAD/min-max normalization.
    """
    if depth_pred.ndim != 3:
        raise ValueError(f"depth_pred must have shape [T,H,W], got {tuple(depth_pred.shape)}")
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [T,3,H,W], got {tuple(rgb.shape)}")
    if depth_pred.shape[0] != rgb.shape[0] or depth_pred.shape[-2:] != rgb.shape[-2:]:
        raise ValueError(
            f"depth_pred and rgb temporal/spatial shapes must match, got "
            f"{tuple(depth_pred.shape)} and {tuple(rgb.shape)}"
        )
    if isinstance(beta, bool) or not math.isfinite(float(beta)) or beta < 0:
        raise ValueError(f"beta must be a finite non-negative number, got {beta!r}")

    device = depth_pred.device
    depth_pred = depth_pred.float()
    rgb = rgb.to(device=device, dtype=depth_pred.dtype)

    if eval_mask is None:
        raise ValueError(
            "CAPA strict OPW requires eval_mask=depth_gt_valid to define Omega"
        )
    if eval_mask.shape != depth_pred.shape:
        raise ValueError(
            "eval_mask must have the same [T,H,W] shape as depth_pred, got "
            f"{tuple(eval_mask.shape)} and {tuple(depth_pred.shape)}"
        )
    eval_mask = eval_mask.to(device=device, dtype=torch.bool)
    if flow_batch_size is not None and flow_batch_size <= 0:
        raise ValueError(
            f"flow_batch_size must be positive when provided, got {flow_batch_size}"
        )
    if flow_max_side is not None and flow_max_side <= 0:
        raise ValueError(
            f"flow_max_side must be positive when provided, got {flow_max_side}"
        )

    finite_rgb = rgb[torch.isfinite(rgb)]
    if finite_rgb.numel() > 0 and finite_rgb.max() > 2:
        rgb = rgb / 255.0

    T, H, W = depth_pred.shape
    if T < 2:
        opw = float("nan")
        if return_details:
            return opw, {
                "per_pair_opw": torch.empty(0, dtype=depth_pred.dtype),
                "valid_count": torch.empty(0, dtype=torch.long),
                "weight_sum": torch.empty(0, dtype=depth_pred.dtype),
                "flow_valid_count": torch.empty(0, dtype=torch.long),
                "valid_weight_count": torch.empty(0, dtype=torch.long),
                "invalid_correspondence_count": torch.empty(0, dtype=torch.long),
                "invalid_warped_depth_count": torch.empty(0, dtype=torch.long),
                "beta": float(beta),
                "protocol": OPW_PROTOCOL,
                "fb_consistency": bool(fb_consistency),
            }
        return opw

    if flow_model is None:
        flow_model = load_gmflow(device=device)

    rgb_clean = torch.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_finite = torch.isfinite(rgb).all(dim=1)

    src_rgb = rgb_clean[:-1]
    tgt_rgb = rgb_clean[1:]
    src_depth = depth_pred[:-1]
    tgt_depth = depth_pred[1:]
    omega = eval_mask[1:]

    backward_flow = _run_flow_model(
        flow_model,
        tgt_rgb,
        src_rgb,
        batch_size=flow_batch_size,
        max_side=flow_max_side,
    )
    if backward_flow.shape != (T - 1, 2, H, W):
        raise RuntimeError(
            f"Backward flow must have shape {(T - 1, 2, H, W)}, got {tuple(backward_flow.shape)}"
        )

    # Metric depth must be finite and strictly positive. CAPA Eq. (10) states
    # that invalid warped depth receives zero visibility weight.
    source_depth_valid = torch.isfinite(src_depth) & (src_depth > 0)
    current_depth_valid = torch.isfinite(tgt_depth)
    source_depth_clean = torch.where(source_depth_valid, src_depth, torch.zeros_like(src_depth))

    warped_depth, flow_valid = bilinear_warp_by_backward_flow(
        source_depth_clean.unsqueeze(1), backward_flow
    )
    warped_depth = warped_depth.squeeze(1)
    warped_source_depth_valid, _ = bilinear_warp_by_backward_flow(
        source_depth_valid.float().unsqueeze(1), backward_flow
    )
    warped_source_depth_valid = warped_source_depth_valid.squeeze(1) > 1.0 - 1e-6

    warped_rgb, rgb_flow_valid = bilinear_warp_by_backward_flow(src_rgb, backward_flow)
    warped_source_rgb_valid, _ = bilinear_warp_by_backward_flow(
        rgb_finite[:-1].float().unsqueeze(1), backward_flow
    )
    warped_source_rgb_valid = warped_source_rgb_valid.squeeze(1) > 1.0 - 1e-6

    valid_weight = (
        flow_valid
        & rgb_flow_valid
        & warped_source_depth_valid
        & warped_source_rgb_valid
        & current_depth_valid
        & rgb_finite[1:]
        & torch.isfinite(warped_depth)
        & torch.isfinite(warped_rgb).all(dim=1)
    )

    if fb_consistency:
        forward_flow = _run_flow_model(
            flow_model,
            src_rgb,
            tgt_rgb,
            batch_size=flow_batch_size,
            max_side=flow_max_side,
        )
        if forward_flow.shape != (T - 1, 2, H, W):
            raise RuntimeError(
                f"Forward flow must have shape {(T - 1, 2, H, W)}, got {tuple(forward_flow.shape)}"
            )
        valid_weight = valid_weight & _forward_backward_consistency_mask(
            backward_flow, forward_flow
        )

    rgb_err = ((tgt_rgb - warped_rgb) ** 2).sum(dim=1)
    weight = torch.exp(-float(beta) * rgb_err)
    weight = torch.where(valid_weight, weight, torch.zeros_like(weight))

    depth_abs_diff = torch.abs(tgt_depth - warped_depth)
    depth_abs_diff = torch.where(
        torch.isfinite(depth_abs_diff), depth_abs_diff, torch.zeros_like(depth_abs_diff)
    )

    # CAPA Eq. (9): Omega is the valid dense-GT mask in the target frame. It
    # controls both the summation domain and the unweighted denominator. Flow,
    # warped-depth, and RGB validity only zero the visibility weight (Eq. 10).
    valid_count = omega.sum(dim=(-1, -2))
    numerator = (weight * depth_abs_diff * omega.to(weight.dtype)).sum(dim=(-1, -2))
    denominator = valid_count.clamp(min=1).to(dtype=depth_pred.dtype)
    pair_opw = numerator / denominator
    pair_opw = torch.where(
        valid_count > 0,
        pair_opw,
        torch.full_like(pair_opw, float("nan")),
    )

    opw_tensor = pair_opw.mean() * 100.0

    opw = float(opw_tensor.detach().cpu().item())
    if return_details:
        flow_valid_eval = flow_valid & rgb_flow_valid & omega
        valid_weight_eval = valid_weight & omega
        invalid_correspondence = omega & ~(flow_valid & rgb_flow_valid)
        invalid_warped_depth = (
            omega & (flow_valid & rgb_flow_valid) & ~warped_source_depth_valid
        )
        details: dict[str, Any] = {
            "per_pair_opw": (pair_opw.detach().cpu() * 100.0),
            "valid_count": valid_count.detach().cpu(),
            "weight_sum": (weight * omega.to(weight.dtype))
            .sum(dim=(-1, -2))
            .detach()
            .cpu(),
            "flow_valid_count": flow_valid_eval.sum(dim=(-1, -2)).detach().cpu(),
            "valid_weight_count": valid_weight_eval.sum(dim=(-1, -2)).detach().cpu(),
            "invalid_correspondence_count": invalid_correspondence.sum(dim=(-1, -2)).detach().cpu(),
            "invalid_warped_depth_count": (
                invalid_warped_depth.sum(dim=(-1, -2)).detach().cpu()
            ),
            "beta": float(beta),
            "protocol": OPW_PROTOCOL,
            "fb_consistency": bool(fb_consistency),
        }
        return opw, details
    return opw


def abs_relative_difference(
    pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Absolute relative difference: |pred - gt| / gt.  Returns per-pixel map."""
    return torch.abs(pred - gt) / (gt + eps)


def squared_relative_difference(
    pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Squared relative difference: (pred - gt)^2 / gt.  Returns per-pixel map."""
    return torch.pow(pred - gt, 2) / (gt + eps)


def absrel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean Absolute Relative error (AbsRel).

    Args:
        pred: [B, H, W] or [H, W] predicted depth.
        gt:   [B, H, W] or [H, W] ground-truth depth.
        valid_mask: optional boolean mask.

    Returns:
        Scalar tensor.
    """
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    err = abs_relative_difference(pred, gt, eps=eps)
    return mask_aware_batch_mean(err, valid_mask).mean()


def sqrel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean Squared Relative error (SqRel)."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    err = squared_relative_difference(pred, gt, eps=eps)
    return mask_aware_batch_mean(err, valid_mask).mean()


def rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Root Mean Squared Error (RMSE)."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = pred - gt
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    mse = torch.sum(diff.pow(2), (-1, -2)) / n
    return torch.sqrt(mse).mean()


def rmse_log(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Root Mean Squared Log Error."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = torch.log(pred) - torch.log(gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    mse = torch.sum(diff.pow(2), (-1, -2)) / n
    return torch.sqrt(mse).mean()


def log10_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute log10 error."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    if valid_mask is not None:
        diff = torch.abs(torch.log10(pred[valid_mask]) - torch.log10(gt[valid_mask]))
    else:
        diff = torch.abs(torch.log10(pred) - torch.log10(gt))
    return diff.mean()


def silog(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale-Invariant Logarithmic error (SIlog, in %)."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = torch.log(pred) - torch.log(gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = gt.shape[-2] * gt.shape[-1]
    first_term = torch.sum(diff.pow(2), (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    return torch.sqrt(torch.mean(first_term - second_term)) * 100


def irmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse RMSE (iRMSE): RMSE computed on inverse depth."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = (1.0 / pred) - (1.0 / gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    mse = torch.sum(diff.pow(2), (-1, -2)) / n
    return torch.sqrt(mse).mean()


def _threshold_acc(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of pixels where max(pred/gt, gt/pred) < threshold."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    d = torch.max(pred / gt, gt / pred)
    bit = (d < threshold).float()
    if valid_mask is not None:
        bit[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    return (torch.sum(bit, (-1, -2)) / n).mean()


def delta1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delta1 accuracy (threshold = 1.25)."""
    return _threshold_acc(pred, gt, 1.25, valid_mask)


def delta2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delta2 accuracy (threshold = 1.25^2)."""
    return _threshold_acc(pred, gt, 1.25**2, valid_mask)


def delta3(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delta3 accuracy (threshold = 1.25^3)."""
    return _threshold_acc(pred, gt, 1.25**3, valid_mask)


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Compute a standard set of depth metrics.

    Args:
        pred: [B, H, W] or [H, W] predicted depth (metric scale, > 0).
        gt:   [B, H, W] or [H, W] ground-truth depth (> 0 where valid).
        valid_mask: optional boolean mask (True = valid).  If *None* and gt
                    contains zeros, pixels with ``gt <= 0`` are auto-masked.

    Returns:
        Dictionary with keys: absrel, sqrel, rmse, rmse_log, log10,
        silog, irmse, delta1, delta2, delta3.
    """
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)

    # Auto-mask: gt must be positive for all ratio-based metrics
    if valid_mask is None:
        valid_mask = gt > 0
    else:
        valid_mask = valid_mask & (gt > 0)

    # Clamp pred to positive for log-based metrics
    pred = pred.clamp(min=1e-6)

    return {
        "absrel": absrel(pred, gt, valid_mask).item(),
        "sqrel": sqrel(pred, gt, valid_mask).item(),
        "rmse": rmse(pred, gt, valid_mask).item(),
        "rmse_log": rmse_log(pred, gt, valid_mask).item(),
        "log10": log10_error(pred, gt, valid_mask).item(),
        "silog": silog(pred, gt, valid_mask).item(),
        "irmse": irmse(pred, gt, valid_mask).item(),
        "delta1": delta1(pred, gt, valid_mask).item(),
        "delta2": delta2(pred, gt, valid_mask).item(),
        "delta3": delta3(pred, gt, valid_mask).item(),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a single-line human-readable summary string."""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)


def average_metrics(
    list_of_dicts: list[dict[str, Any]],
    ignore_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Average numeric values across a list of metric dicts.

    Non-numeric and non-finite values are skipped per key. Keys with no finite
    observations are omitted: absence means "not computed", whereas JSON null
    is reserved for an unavailable value on an individual sample.
    """
    keys_to_ignore = set(ignore_keys) if ignore_keys else set()
    all_keys = sorted({k for d in list_of_dicts for k in d.keys()} - keys_to_ignore)

    result: dict[str, Any] = {}
    for k in all_keys:
        values = [
            float(value)
            for d in list_of_dicts
            if isinstance((value := d.get(k)), (int, float))
            and math.isfinite(float(value))
        ]
        if not values:
            logger.info("Omitting metric key `%s`: no finite values were computed", k)
            continue
        result[k] = sum(values) / len(values)

    return result


def _ensure_batched(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Ensure tensors have a batch dimension [B, H, W]."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)
    if valid_mask is not None and valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
    return pred, gt, valid_mask
