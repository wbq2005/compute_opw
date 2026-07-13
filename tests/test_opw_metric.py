import math

import pytest
import torch

import capa.utils.metric as metric
from capa.utils.metric import bilinear_warp_by_backward_flow, compute_opw
from capa.utils.metric import _forward_backward_consistency_mask, _run_flow_model


class FakeFlowModel(torch.nn.Module):
    """CPU-only fake flow model returning zero flow in [B, 2, H, W]."""

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        B, _, H, W = image0.shape
        return torch.zeros(B, 2, H, W, dtype=image0.dtype, device=image0.device)


class ShapeRecordingFlowModel(torch.nn.Module):
    """Fake flow model that records the padded tensor shape it receives."""

    _capa_pad_to_multiple = 16

    def __init__(self) -> None:
        super().__init__()
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        self.seen_shape = tuple(image0.shape)
        B, _, H, W = image0.shape
        return torch.zeros(B, 2, H, W, dtype=image0.dtype, device=image0.device)


class BatchRecordingFlowModel(torch.nn.Module):
    """Fake flow model that records each inference batch size."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        B, _, H, W = image0.shape
        self.batch_sizes.append(B)
        return torch.zeros(B, 2, H, W, dtype=image0.dtype, device=image0.device)


class UnitFlowRecordingModel(torch.nn.Module):
    """Fake flow model returning one pixel of x/y flow at inference size."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        self.seen_shape = tuple(image0.shape)
        B, _, H, W = image0.shape
        return torch.ones(B, 2, H, W, dtype=image0.dtype, device=image0.device)


class RightSamplingFlowModel(torch.nn.Module):
    """Backward flow that samples one pixel to the right in the source frame."""

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, **kwargs) -> torch.Tensor:
        del image1, kwargs
        B, _, H, W = image0.shape
        flow = torch.zeros(B, 2, H, W, dtype=image0.dtype, device=image0.device)
        flow[:, 0] = 1.0
        return flow


def all_pixels(depth: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(depth, dtype=torch.bool)


def test_resolve_gmflow_paths_finds_standard_checkpoint(tmp_path):
    repo = tmp_path / "gmflow"
    (repo / "gmflow").mkdir(parents=True)
    (repo / "gmflow" / "gmflow.py").touch()
    checkpoint = repo / "pretrained" / "models" / "gmflow_sintel-test.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    resolved_repo, resolved_checkpoint = metric.resolve_gmflow_paths(
        gmflow_repo=repo
    )

    assert resolved_repo == repo.resolve()
    assert resolved_checkpoint == checkpoint.resolve()


def test_compute_opw_auto_loads_gmflow_when_model_is_omitted(monkeypatch):
    depth = torch.stack([torch.ones(2, 3), torch.full((2, 3), 1.1)])
    rgb = torch.zeros(2, 3, 2, 3)
    loaded: list[torch.device] = []

    def fake_load_gmflow(*, device):
        loaded.append(torch.device(device))
        return FakeFlowModel()

    monkeypatch.setattr(metric, "load_gmflow", fake_load_gmflow)

    opw = compute_opw(depth, rgb, eval_mask=all_pixels(depth))

    assert loaded == [torch.device("cpu")]
    assert opw == pytest.approx(10.0, abs=1e-5)


def test_warp_zero_flow_identity():
    H, W = 4, 5
    x = torch.arange(H * W, dtype=torch.float32).view(1, 1, H, W)
    flow = torch.zeros(1, 2, H, W)

    warped, valid_mask = bilinear_warp_by_backward_flow(x, flow)

    assert torch.allclose(warped, x, atol=1e-6)
    assert valid_mask.shape == (1, H, W)
    assert valid_mask.all()


def test_run_flow_model_uses_gmflow_style_padding_and_crops_back():
    image0 = torch.rand(1, 3, 17, 19)
    image1 = torch.rand(1, 3, 17, 19)
    flow_model = ShapeRecordingFlowModel()

    flow = _run_flow_model(flow_model, image0, image1)

    assert flow_model.seen_shape == (1, 3, 32, 32)
    assert flow.shape == (1, 2, 17, 19)


def test_run_flow_model_chunks_pairs_without_changing_output_shape():
    image0 = torch.rand(5, 3, 4, 6)
    image1 = torch.rand(5, 3, 4, 6)
    flow_model = BatchRecordingFlowModel()

    flow = _run_flow_model(flow_model, image0, image1, batch_size=2)

    assert flow_model.batch_sizes == [2, 2, 1]
    assert flow.shape == (5, 2, 4, 6)
    assert torch.count_nonzero(flow) == 0


def test_run_flow_model_resizes_and_scales_flow_back_to_original_grid():
    image0 = torch.rand(1, 3, 8, 6)
    image1 = torch.rand(1, 3, 8, 6)
    flow_model = UnitFlowRecordingModel()

    flow = _run_flow_model(flow_model, image0, image1, max_side=4)

    assert flow_model.seen_shape == (1, 3, 4, 3)
    assert flow.shape == (1, 2, 8, 6)
    assert torch.allclose(flow[:, 0], torch.full_like(flow[:, 0], 2.0))
    assert torch.allclose(flow[:, 1], torch.full_like(flow[:, 1], 2.0))


def test_compute_opw_flow_chunking_preserves_metric():
    T, H, W = 6, 4, 5
    depth = torch.arange(T, dtype=torch.float32).view(T, 1, 1).expand(T, H, W)
    rgb = torch.zeros(T, 3, H, W)
    eval_mask = all_pixels(depth)
    flow_model = BatchRecordingFlowModel()

    chunked = compute_opw(
        depth,
        rgb,
        flow_model,
        eval_mask=eval_mask,
        flow_batch_size=2,
    )
    unchunked = compute_opw(depth, rgb, FakeFlowModel(), eval_mask=eval_mask)

    assert flow_model.batch_sizes == [2, 2, 1]
    assert chunked == pytest.approx(unchunked, abs=1e-6)


def test_fb_consistency_threshold_matches_gmflow_definition():
    backward = torch.zeros(1, 2, 2, 3)
    forward = torch.zeros_like(backward)
    backward[:, 0] = 1.0
    forward[:, 0, :, 0] = -100.0
    forward[:, 0, :, 1] = -2.0
    forward[:, 0, :, 2] = -1.0

    mask = _forward_backward_consistency_mask(backward, forward)

    # Backward flow samples forward one pixel to the right. At x=0 the error
    # is 1.0. GMFlow uses the unwarped |forward[x=0]|=100 in the threshold,
    # so this correspondence is retained; using warped_forward would reject it.
    assert mask[0, :, 0].all()
    assert not mask[0, :, -1].any()  # backward sample is out of bounds


def test_compute_opw_identical_frames_zero():
    T, H, W = 3, 4, 5
    depth = torch.ones(T, H, W)
    rgb_frame = torch.rand(3, H, W)
    rgb = rgb_frame.unsqueeze(0).repeat(T, 1, 1, 1)

    opw = compute_opw(depth, rgb, FakeFlowModel(), eval_mask=all_pixels(depth))

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

    opw = compute_opw(depth, rgb, FakeFlowModel(), eval_mask=all_pixels(depth))

    assert opw == pytest.approx(10.0, abs=1e-5)


def test_compute_opw_uses_target_to_source_backward_flow():
    source_depth = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    target_depth = torch.tensor([[2.0, 3.0, 99.0], [2.0, 3.0, 99.0]])
    depth = torch.stack([source_depth, target_depth])

    source_rgb = torch.tensor([0.0, 0.1, 0.2]).view(1, 1, 3).repeat(3, 2, 1)
    target_rgb = torch.tensor([0.1, 0.2, 0.0]).view(1, 1, 3).repeat(3, 2, 1)
    rgb = torch.stack([source_rgb, target_rgb])

    eval_mask = torch.ones(2, 2, 3, dtype=torch.bool)
    eval_mask[1, :, -1] = False

    opw = compute_opw(
        depth,
        rgb,
        RightSamplingFlowModel(),
        eval_mask=eval_mask,
    )

    assert opw == pytest.approx(0.0, abs=1e-6)


def test_compute_opw_details_include_formula_validity_counts():
    H, W = 4, 5
    depth = torch.ones(2, H, W)
    rgb = torch.zeros(2, 3, H, W)

    opw, details = compute_opw(
        depth,
        rgb,
        FakeFlowModel(),
        eval_mask=all_pixels(depth),
        return_details=True,
    )

    assert opw == pytest.approx(0.0, abs=1e-6)
    assert details["valid_count"].tolist() == [H * W]
    assert details["flow_valid_count"].tolist() == [H * W]
    assert details["valid_weight_count"].tolist() == [H * W]
    assert details["invalid_correspondence_count"].tolist() == [0]
    assert details["invalid_warped_depth_count"].tolist() == [0]
    assert details["protocol"] == "capa_strict"


def test_compute_opw_uses_dense_gt_mask_for_omega():
    H, W = 4, 5
    depth_0 = torch.ones(H, W)
    depth_1 = torch.full((H, W), 1.1)
    depth_1[0, 0] = 9.0
    depth = torch.stack([depth_0, depth_1])
    rgb = torch.zeros(2, 3, H, W)
    eval_mask = torch.ones(2, H, W, dtype=torch.bool)
    eval_mask[1, 0, 0] = False

    opw, details = compute_opw(
        depth,
        rgb,
        FakeFlowModel(),
        eval_mask=eval_mask,
        return_details=True,
    )

    assert opw == pytest.approx(10.0, abs=1e-5)
    assert details["valid_count"].tolist() == [H * W - 1]


def test_compute_opw_invalid_warped_depth_has_zero_weight_but_stays_in_denominator():
    H, W = 4, 5
    depth_0 = torch.ones(H, W)
    depth_0[0, 0] = float("nan")
    depth_1 = torch.full((H, W), 1.1)
    depth = torch.stack([depth_0, depth_1])
    rgb = torch.zeros(2, 3, H, W)
    eval_mask = torch.ones(2, H, W, dtype=torch.bool)

    opw, details = compute_opw(
        depth,
        rgb,
        FakeFlowModel(),
        eval_mask=eval_mask,
        return_details=True,
    )

    expected = 100.0 * 0.1 * (H * W - 1) / (H * W)
    assert opw == pytest.approx(expected, abs=1e-5)
    assert details["valid_count"].tolist() == [H * W]
    assert details["invalid_warped_depth_count"].tolist() == [1]


def test_compute_opw_nonpositive_warped_metric_depth_is_invalid():
    H, W = 4, 5
    depth_0 = torch.ones(H, W)
    depth_0[0, 0] = -1.0
    depth_1 = torch.full((H, W), 1.1)
    depth = torch.stack([depth_0, depth_1])
    rgb = torch.zeros(2, 3, H, W)
    eval_mask = torch.ones(2, H, W, dtype=torch.bool)

    opw, details = compute_opw(
        depth,
        rgb,
        FakeFlowModel(),
        eval_mask=eval_mask,
        return_details=True,
    )

    expected = 100.0 * 0.1 * (H * W - 1) / (H * W)
    assert opw == pytest.approx(expected, abs=1e-5)
    assert details["invalid_warped_depth_count"].tolist() == [1]


def test_compute_opw_rgb_weight_is_exp_of_squared_l2_norm():
    H, W = 2, 3
    depth = torch.stack([torch.ones(H, W), torch.full((H, W), 1.2)])
    rgb = torch.zeros(2, 3, H, W)
    rgb[1, 0] = 0.1

    opw = compute_opw(depth, rgb, FakeFlowModel(), eval_mask=all_pixels(depth))

    expected_weight = math.exp(-50.0 * 0.1**2)
    assert opw == pytest.approx(100.0 * 0.2 * expected_weight, abs=1e-5)


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

    eval_mask = all_pixels(depth)
    opw_01 = compute_opw(depth, rgb_01, FakeFlowModel(), eval_mask=eval_mask)
    opw_255 = compute_opw(depth, rgb_255, FakeFlowModel(), eval_mask=eval_mask)

    assert opw_255 == pytest.approx(opw_01, abs=1e-5)


def test_eval_mask_shape_must_match_prediction():
    H, W = 4, 5
    depth = torch.ones(2, H, W)
    rgb = torch.zeros(2, 3, H, W)
    eval_mask = torch.ones(2, H - 1, W, dtype=torch.bool)

    with pytest.raises(ValueError, match="eval_mask"):
        compute_opw(depth, rgb, FakeFlowModel(), eval_mask=eval_mask)


def test_compute_opw_requires_explicit_omega():
    depth = torch.ones(2, 2, 3)
    rgb = torch.zeros(2, 3, 2, 3)

    with pytest.raises(ValueError, match="requires eval_mask"):
        compute_opw(depth, rgb, FakeFlowModel())


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), -1.0, True])
def test_compute_opw_rejects_invalid_beta(beta):
    depth = torch.ones(2, 2, 3)
    rgb = torch.zeros(2, 3, 2, 3)

    with pytest.raises(ValueError, match="beta must be"):
        compute_opw(
            depth,
            rgb,
            FakeFlowModel(),
            beta=beta,
            eval_mask=all_pixels(depth),
        )


def test_short_sequence():
    # Documented behavior: a sequence with fewer than two frames has no adjacent
    # pair to evaluate, so compute_opw returns NaN instead of raising.
    H, W = 4, 5
    depth = torch.ones(1, H, W)
    rgb = torch.zeros(1, 3, H, W)

    opw, details = compute_opw(
        depth,
        rgb,
        FakeFlowModel(),
        eval_mask=all_pixels(depth),
        return_details=True,
    )

    assert math.isnan(opw)
    assert details["per_pair_opw"].numel() == 0
    assert details["valid_count"].numel() == 0
    assert details["weight_sum"].numel() == 0
