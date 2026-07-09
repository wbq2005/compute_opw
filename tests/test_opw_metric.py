import math

import pytest
import torch

from capa.utils.metric import bilinear_warp_by_backward_flow, compute_opw


class FakeFlowModel(torch.nn.Module):
    """CPU-only fake flow model returning zero flow in [B, 2, H, W]."""

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        B, _, H, W = image0.shape
        return torch.zeros(B, 2, H, W, dtype=image0.dtype, device=image0.device)


def test_warp_zero_flow_identity():
    H, W = 4, 5
    x = torch.arange(H * W, dtype=torch.float32).view(1, 1, H, W)
    flow = torch.zeros(1, 2, H, W)

    warped, valid_mask = bilinear_warp_by_backward_flow(x, flow)

    assert torch.allclose(warped, x, atol=1e-6)
    assert valid_mask.shape == (1, H, W)
    assert valid_mask.all()


def test_compute_opw_identical_frames_zero():
    T, H, W = 3, 4, 5
    depth = torch.ones(T, H, W)
    rgb_frame = torch.rand(3, H, W)
    rgb = rgb_frame.unsqueeze(0).repeat(T, 1, 1, 1)

    opw = compute_opw(depth, rgb, FakeFlowModel())

    assert opw == pytest.approx(0.0, abs=1e-6)


def test_compute_opw_constant_depth_offset():
    H, W = 4, 5
    depth = torch.stack(
        [
            torch.full((H, W), 1.0),
            torch.full((H, W), 1.1),
        ],
        dim=0,
    )
    rgb = torch.zeros(2, 3, H, W)

    opw = compute_opw(depth, rgb, FakeFlowModel())

    assert opw == pytest.approx(10.0, abs=1e-5)


def test_rgb_uint8_auto_scaling():
    T, H, W = 2, 4, 5
    depth = torch.stack(
        [
            torch.full((H, W), 1.0),
            torch.full((H, W), 1.2),
        ],
        dim=0,
    )
    rgb_01 = torch.rand(T, 3, H, W)
    rgb_255 = rgb_01 * 255.0

    opw_01 = compute_opw(depth, rgb_01, FakeFlowModel())
    opw_255 = compute_opw(depth, rgb_255, FakeFlowModel())

    assert opw_255 == pytest.approx(opw_01, abs=1e-5)


def test_invalid_depth_mask():
    H, W = 4, 5
    depth_0 = torch.ones(H, W)
    depth_1 = torch.full((H, W), 1.1)
    depth_1[0, 0] = 0.0
    depth_1[0, 1] = float("nan")
    depth_1[0, 2] = float("inf")
    depth = torch.stack([depth_0, depth_1], dim=0)
    rgb = torch.zeros(2, 3, H, W)

    opw, details = compute_opw(depth, rgb, FakeFlowModel(), return_details=True)

    assert math.isfinite(opw)
    assert details["valid_count"].tolist() == [H * W - 3]


def test_short_sequence():
    # Documented behavior: a sequence with fewer than two frames has no adjacent
    # pair to evaluate, so compute_opw returns NaN instead of raising.
    H, W = 4, 5
    depth = torch.ones(1, H, W)
    rgb = torch.zeros(1, 3, H, W)

    opw, details = compute_opw(depth, rgb, FakeFlowModel(), return_details=True)

    assert math.isnan(opw)
    assert details["per_pair_opw"].numel() == 0
    assert details["valid_count"].numel() == 0
    assert details["weight_sum"].numel() == 0
