# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
Depth alignment utilities for CAPA.

Solves optimal scale and shift to align predicted depth to condition depth
using robust L1 optimization (weighted median).

Adapted from MoGe alignment: https://github.com/microsoft/MoGe
"""

from typing import Optional, Tuple, Union
import math
import torch


def scatter_min(
    size: int, dim: int, index: torch.LongTensor, src: torch.Tensor
) -> torch.return_types.min:
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1 :]
    minimum = torch.full(shape, float("inf"), dtype=src.dtype, device=src.device).scatter_reduce(
        dim=dim, index=index, src=src, reduce="amin", include_self=False
    )
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1 :])] = (
        minimum_where[dim]
    )
    return torch.return_types.min((minimum, indices))


def _pad_inf(x_: torch.Tensor):
    return torch.cat(
        [
            torch.full_like(x_[..., :1], -torch.inf),
            x_,
            torch.full_like(x_[..., :1], torch.inf),
        ],
        dim=-1,
    )


def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)


def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return (
        a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)
    )


def align(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    trunc: Optional[Union[float, torch.Tensor]] = None,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    Solve `min sum_i w_i * |a * x_i - y_i|` (or truncated variant).

    Args:
        x: shape (..., n)
        y: shape (..., n)
        w: shape (..., n), must be >= 0
        trunc: optional truncation threshold

    Returns:
        a: optimal scale, shape (...)
        loss: objective value at a, shape (...)
        index: which anchor was selected, shape (...)
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(
            derivatives, torch.zeros_like(derivatives[..., :1]), side="left"
        ).clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
    else:
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = (
            x.reshape(-1, x.shape[-1]),
            y.reshape(-1, y.shape[-1]),
            w.reshape(-1, w.shape[-1]),
        )

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)

        y_div_x = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            A_sorted, A_argsort = y_div_x.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A_sorted, Q_A = _pad_inf(A_sorted), _pad_cumsum(Q_A)

            B_sorted, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B_sorted, Q_B = _pad_inf(B_sorted), _pad_cumsum(Q_B)

            C_sorted, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C_sorted, Q_C = _pad_inf(C_sorted), _pad_cumsum(Q_C)

            j_A = torch.searchsorted(A_sorted, y_div_x, side="left").sub_(1)
            j_B = torch.searchsorted(B_sorted, y_div_x, side="left").sub_(1)
            j_C = torch.searchsorted(C_sorted, y_div_x, side="left").sub_(1)
            left_derivative = (
                2 * torch.gather(Q_A, dim=-1, index=j_A)
                - torch.gather(Q_B, dim=-1, index=j_B)
                - torch.gather(Q_C, dim=-1, index=j_C)
            )
            j_A = torch.searchsorted(A_sorted, y_div_x, side="right").sub_(1)
            j_B = torch.searchsorted(B_sorted, y_div_x, side="right").sub_(1)
            j_C = torch.searchsorted(C_sorted, y_div_x, side="right").sub_(1)
            right_derivative = (
                2 * torch.gather(Q_A, dim=-1, index=j_A)
                - torch.gather(Q_B, dim=-1, index=j_B)
                - torch.gather(Q_C, dim=-1, index=j_C)
            )

            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)

            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]
            MAX_ELEMENTS = 4096**2
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat(
                [
                    _compute_residual(ea[:, None], xyw[ei, :, :], trunc)
                    for ea, ei in zip(
                        extrema_a.split(SPLIT_SIZE),
                        where_extrema_batch.split(SPLIT_SIZE),
                    )
                ]
            )

            minima, indices = scatter_min(
                size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value
            )
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(
            x, dim=-1, index=index[..., None]
        ).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_depth_affine(
    depth_src: torch.Tensor,
    depth_tgt: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find optimal scale and shift: depth_aligned = scale * depth_src + shift.

    Args:
        depth_src: shape (..., N)
        depth_tgt: shape (..., N)
        weight: shape (..., N)

    Returns:
        scale: shape (...)
        shift: shape (...)
    """
    batch_shape, n = depth_src.shape[:-1], depth_src.shape[-1]
    batch_size = math.prod(batch_shape)
    depth_src = depth_src.reshape(batch_size, n)
    depth_tgt = depth_tgt.reshape(batch_size, n)
    weight = weight.reshape(batch_size, n)

    anchors_where_batch, anchors_where_n = torch.where(weight > 0)

    with torch.no_grad():
        depth_src_anchor = depth_src[anchors_where_batch, anchors_where_n]
        depth_tgt_anchor = depth_tgt[anchors_where_batch, anchors_where_n]

        depth_src_anchored = depth_src[anchors_where_batch, :] - depth_src_anchor[..., None]
        depth_tgt_anchored = depth_tgt[anchors_where_batch, :] - depth_tgt_anchor[..., None]
        weight_anchored = weight[anchors_where_batch, :]

        scale, loss, index = align(depth_src_anchored, depth_tgt_anchored, weight_anchored)
        loss, index_anchor = scatter_min(
            size=batch_size, dim=0, index=anchors_where_batch, src=loss
        )

    index_1 = anchors_where_n[index_anchor]
    index_2 = index[index_anchor]

    tgt_1 = torch.gather(depth_tgt, dim=1, index=index_1[..., None]).squeeze(-1)
    src_1 = torch.gather(depth_src, dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2 = torch.gather(depth_tgt, dim=1, index=index_2[..., None]).squeeze(-1)
    src_2 = torch.gather(depth_src, dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1e-7)
    shift = tgt_1 - scale * src_1

    return scale.reshape(batch_shape), shift.reshape(batch_shape)


def align_depth_to_condition(
    depth_pred_nhw: torch.Tensor,
    depth_cond_nhw: torch.Tensor,
    mask_nhw: torch.Tensor,
    n_points: int = 12000,
    seed: int = 47,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align predicted depth to condition depth per-image using affine transform.

    Args:
        depth_pred_nhw: predicted depth [N, H, W]
        depth_cond_nhw: condition (sparse) depth [N, H, W]
        mask_nhw: valid mask for condition depth [N, H, W]
        n_points: max points to use for alignment (for memory efficiency)
        seed: random seed for point subsampling

    Returns:
        scale: [N]
        shift: [N]
    """
    N = depth_pred_nhw.shape[0]
    device = depth_pred_nhw.device
    generator = torch.Generator(device=device).manual_seed(seed)

    scale = torch.full((N,), torch.nan, device=device, dtype=depth_pred_nhw.dtype)
    shift = torch.full((N,), torch.nan, device=device, dtype=depth_pred_nhw.dtype)

    for i in range(N):
        mask_i = mask_nhw[i]
        src = depth_pred_nhw[i][mask_i]
        tgt = depth_cond_nhw[i][mask_i]

        # Subsample for memory
        if src.shape[0] > n_points:
            idx = torch.randperm(src.shape[0], generator=generator, device=device)[:n_points]
            src, tgt = src[idx], tgt[idx]

        s, t = align_depth_affine(
            depth_src=src.unsqueeze(0),
            depth_tgt=tgt.unsqueeze(0),
            weight=torch.ones_like(tgt).unsqueeze(0),
        )
        scale[i] = s.squeeze()
        shift[i] = t.squeeze()

    assert not torch.any(torch.isnan(scale))
    assert not torch.any(torch.isnan(shift))
    return scale, shift
