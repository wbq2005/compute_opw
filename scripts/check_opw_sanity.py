#!/usr/bin/env python
"""Local CPU-only OPW sanity checks with synthetic data and fake zero flow.

Usage:
    python scripts/check_opw_sanity.py

This script intentionally does not import GMFlow, load checkpoints, use CUDA, or
touch any dataset files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capa.utils.metric import compute_opw  # noqa: E402


class FakeFlowModel(torch.nn.Module):
    """Return zero optical flow with shape [B, 2, H, W]."""

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        B, _, H, W = image0.shape
        return torch.zeros(B, 2, H, W, dtype=image0.dtype, device=image0.device)


def _check_close(name: str, observed: float, expected: float, tol: float = 1e-5) -> bool:
    passed = abs(observed - expected) <= tol
    status = "PASS" if passed else "FAIL"
    print(f"{name}\tobserved={observed:.8f}\texpected={expected:.8f}\t{status}")
    return passed


def main() -> int:
    flow_model = FakeFlowModel()
    all_passed = True

    # Case 1: identical depth/rgb should have zero OPW.
    T, H, W = 3, 4, 5
    depth = torch.ones(T, H, W)
    rgb_frame = torch.rand(3, H, W)
    rgb = rgb_frame.unsqueeze(0).repeat(T, 1, 1, 1)
    observed = compute_opw(depth, rgb, flow_model)
    all_passed &= _check_close("identical_depth_rgb", observed, 0.0)

    # Case 2: zero flow and identical RGB, D1-D0 = 0.1m => reported OPW = 0.1 * 100 = 10.
    depth = torch.stack(
        [
            torch.full((H, W), 1.0),
            torch.full((H, W), 1.1),
        ],
        dim=0,
    )
    rgb = torch.zeros(2, 3, H, W)
    observed = compute_opw(depth, rgb, flow_model)
    all_passed &= _check_close("constant_depth_offset_0.1", observed, 10.0)

    # Case 3: RGB [0, 255] should be auto-scaled to match RGB [0, 1].
    depth = torch.stack(
        [
            torch.full((H, W), 1.0),
            torch.full((H, W), 1.2),
        ],
        dim=0,
    )
    rgb_01 = torch.rand(2, 3, H, W)
    rgb_255 = rgb_01 * 255.0
    expected = compute_opw(depth, rgb_01, flow_model)
    observed = compute_opw(depth, rgb_255, flow_model)
    all_passed &= _check_close("rgb_uint8_auto_scaling", observed, expected)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
