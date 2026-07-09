# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""
Unified CAPA protocol: test-time adaptation via LoRA or VPT for depth completion.

This module implements the core optimization loop that:
1. Injects adaptation parameters (LoRA or VPT) into a frozen depth model
2. Optimizes these parameters using sparse depth conditions
3. Produces completed depth maps
"""

import math
from pathlib import Path

import torch

_CAPA_ROOT = Path(__file__).parent.parent  # repo root containing assets/

from .utils.alignment import align_depth_to_condition
from .utils.logging import get_local_logger
from .model_wrappers import get_model, BaseModel

logger = get_local_logger(__name__)


class CAPAProtocol:
    """
    CAPA: depth Completion As Parameter-efficient Adaptation.

    Optimizes lightweight adaptation parameters (LoRA or VPT tokens) on a
    frozen depth model, using sparse depth conditions as supervision.

    Args:
        config: Configuration dictionary (loaded from YAML)
        device: torch device
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        # Load model
        self.model: BaseModel = get_model(config["model_name"])
        self.model.load(config["ckpt_path"], device)

        # Tuning settings
        self.tuning_mode: str = config["tuning_mode"]  # "lora" or "vpt"
        self.n_steps: int = config["n_steps"]
        self.lr: float = config["lr"]
        self.gradient_clip_norm: float = config.get("gradient_clip_norm", 1.0)
        self.max_bs_optimize: int = config.get("max_bs_optimize", 10)
        self.max_bs_infer: int | None = config.get("max_bs_infer", None)
        self.pct_random_key_frame: int = config.get("pct_random_key_frame", 10)
        self.max_nan_retries: int = config.get("max_nan_retries", 0)

        # LR scheduler
        self.lr_scheduler_cfg = config.get("lr_scheduler", None)

        # Alignment settings
        align_cfg = config.get("alignment", {})
        self.align_n_points: int = align_cfg.get("n_points", 12000)
        self.align_seed: int = align_cfg.get("seed", 47)

    def run(
        self,
        rgb_n3hw: torch.Tensor,
        depth_cond_nhw: torch.Tensor,
        mask_cond_nhw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run CAPA depth completion on a scene.

        Args:
            rgb_n3hw: RGB frames [N, 3, H, W] in [0, 1]
            depth_cond_nhw: sparse condition depth [N, H, W]
            mask_cond_nhw: mask for valid condition pixels [N, H, W] (bool)

        Returns:
            depth_pred_nhw: completed depth map [N, H, W]
        """
        for attempt in range(self.max_nan_retries + 1):
            # This function will retry if NaN is detected in the model's depth predictions
            # serving as a dirty fix to UniDepth's (rare) NaN issue on optimization.
            result = self._run_optimization(rgb_n3hw, depth_cond_nhw, mask_cond_nhw)
            if result is not None:
                return result
            logger.warning(
                f"NaN detected in depth prediction, retrying ({attempt + 1}/{self.max_nan_retries})"
            )
        raise RuntimeError(
            f"Depth prediction still contains NaN after {self.max_nan_retries} retries"
        )

    def _run_optimization(
        self,
        rgb_n3hw: torch.Tensor,
        depth_cond_nhw: torch.Tensor,
        mask_cond_nhw: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Core optimization loop.  Returns None if NaN is detected in the
        model's depth predictions so the caller can retry.
        """
        device = self.device
        N, _, H, W = rgb_n3hw.shape

        # --- 1. Inject adaptation parameters ---
        self.model.reset()
        if self.tuning_mode == "lora":
            self.model.inject_lora(
                rank=self.config["lora_rank"],
                alpha=self.config["lora_alpha"],
                target_modules=self.config["lora_target_modules"],
            )
        elif self.tuning_mode == "vpt":
            init_method = self.config.get("vpt_init_method", "xavier")
            if init_method.startswith("load_"):
                rel_path = init_method[len("load_") :]
                init_method = f"load_{_CAPA_ROOT / rel_path}"
            self.model.inject_vpt(
                token_len=self.config["vpt_token_len"],
                init_method=init_method,
            )
        else:
            raise ValueError(f"Unknown tuning mode: {self.tuning_mode}")

        # --- 2. Setup optimizer ---
        param_groups = self.model.get_trainable_params(lr=self.lr)
        optimizer = torch.optim.AdamW(param_groups)
        optimizer.zero_grad()

        # LR scheduler
        lr_scheduler = None
        if self.lr_scheduler_cfg is not None:
            lr_scheduler_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler_cfg["type"])
            lr_scheduler = lr_scheduler_cls(optimizer, **self.lr_scheduler_cfg["kwargs"])

        preprocess_inputs = getattr(self.model, "preprocess_inputs", None)
        predict_depth_from_processed = getattr(self.model, "predict_depth_from_processed", None)
        rgb_processed_n3hw = None
        if callable(preprocess_inputs) and callable(predict_depth_from_processed):
            rgb_processed_n3hw = preprocess_inputs(rgb_n3hw.contiguous().to(device))

        # --- 3. Optimization loop ---
        frame_sampling_rng = torch.Generator(device="cpu").manual_seed(self.align_seed)

        for step in range(self.n_steps + 1):  # n_steps optimization + 1 final inference
            is_last_step = step == self.n_steps

            # Sample frames
            if is_last_step or self.pct_random_key_frame <= 0:
                frame_idxs = list(range(N))
            else:
                n_sample = max(1, math.ceil(N * 0.01 * self.pct_random_key_frame))
                frame_idxs = torch.randperm(N, generator=frame_sampling_rng)[:n_sample].tolist()

            # Split into sub-batches.  At the final inference step we use max_bs_infer
            # (None = no limit, all frames in one pass to preserve full cross-frame
            # attention as in the original VGGT_PT protocol).
            if is_last_step:
                if self.max_bs_infer is None:
                    sub_batches = [frame_idxs]
                else:
                    sub_batches = self._make_sub_batches(frame_idxs, self.max_bs_infer)
            else:
                sub_batches = self._make_sub_batches(frame_idxs, self.max_bs_optimize)

            # Output buffer for last step
            if is_last_step:
                depth_pred_nhw = torch.full((N, H, W), torch.nan, device=device)

            loss_accum = 0.0

            for sub_idxs in sub_batches:
                if rgb_processed_n3hw is None:
                    rgb_sub = rgb_n3hw[sub_idxs].contiguous().to(device)
                else:
                    rgb_sub = rgb_processed_n3hw[sub_idxs].clone().contiguous()
                depth_cond_sub = depth_cond_nhw[sub_idxs].contiguous().to(device)
                mask_cond_sub = mask_cond_nhw[sub_idxs].contiguous().to(device)

                grad_ctx = torch.no_grad() if is_last_step else torch.enable_grad()

                with grad_ctx:
                    if rgb_processed_n3hw is None:
                        depth_pred_sub = self.model.predict_depth(rgb_sub)
                    else:
                        depth_pred_sub = predict_depth_from_processed(
                            rgb_sub,
                            output_size=(H, W),
                        )

                if torch.any(torch.isnan(depth_pred_sub)):
                    return None

                # Align to condition
                with torch.no_grad():
                    scale, shift = align_depth_to_condition(
                        depth_pred_sub.detach(),
                        depth_cond_sub.detach(),
                        mask_cond_sub,
                        n_points=self.align_n_points,
                        seed=self.align_seed,
                    )
                scale, shift = scale.detach(), shift.detach()
                depth_pred_sub = depth_pred_sub * scale.view(-1, 1, 1) + shift.view(-1, 1, 1)

                if is_last_step:
                    depth_pred_nhw[sub_idxs] = depth_pred_sub.detach()  # pyright: ignore[reportPossiblyUnboundVariable]
                    continue

                # L1 loss on masked pixels
                loss = torch.abs(depth_pred_sub - depth_cond_sub)[mask_cond_sub].mean()
                scaled_loss = loss / len(sub_batches)
                scaled_loss.backward()

                loss_accum += loss.item()

            if is_last_step:
                break

            # Optimizer step
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), max_norm=self.gradient_clip_norm
                )
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

            torch.cuda.empty_cache()

            if step % 10 == 0 or step == self.n_steps - 1:
                current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else self.lr
                logger.debug(
                    f"step {(step):3d} - loss={loss_accum / len(sub_batches):.4f}, lr={current_lr:.2e}"
                )

        assert not torch.any(torch.isnan(depth_pred_nhw)), "Depth prediction contains NaN"  # pyright: ignore[reportPossiblyUnboundVariable]
        return depth_pred_nhw  # pyright: ignore[reportPossiblyUnboundVariable]

    @staticmethod
    def _make_sub_batches(frame_idxs: list[int], max_bs: int) -> list[list[int]]:
        """Split frame indices into sub-batches of at most max_bs."""
        if max_bs >= len(frame_idxs):
            return [frame_idxs]
        return [frame_idxs[i : i + max_bs] for i in range(0, len(frame_idxs), max_bs)]
