# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""MoGe-2 model wrapper for CAPA."""

import sys
import os
import torch
from peft import LoraConfig, TaskType, get_peft_model

from .base import BaseModel
from ..utils.logging import get_local_logger

logger = get_local_logger(__name__)

# Add base_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "MoGe_VPT"))
from moge.model.v2 import MoGeModel  # noqa: E402


class MoGeV2Model(BaseModel):
    """MoGe-2 depth estimation model with LoRA/VPT support."""

    def __init__(self):
        self.model: MoGeModel = None
        self.device = None
        self._tuning_mode = None

    def load(self, ckpt_path: str, device: torch.device) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = MoGeModel.from_pretrained(ckpt_path).eval().to(device)
        self._cache_original_model()
        logger.info(f"MoGe-2 model loaded from {ckpt_path}")

    def predict_depth(self, rgb_b3hw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run MoGe-2 depth prediction.

        Args:
            rgb_b3hw: [B, 3, H, W] in [0, 1], on self.device

        Returns:
            depth_bhw: [B, H, W]
        """
        use_prompt = self._tuning_mode == "vpt"
        output = self.model.infer_naive_prompt(
            image=rgb_b3hw,
            apply_mask=False,
            use_prompt=use_prompt,
            cast_dtype_vit=torch.bfloat16,
            cast_dtype_head=torch.float32,
        )
        # Depth = z-coordinate of points * metric scale
        points_bhw3 = output["points_raw"] * output["metric_scale"].view(-1, 1, 1, 1)
        depth_bhw = points_bhw3[..., 2]
        return depth_bhw

    def inject_lora(self, rank: int, alpha: int, target_modules: list[str]) -> None:
        self._tuning_mode = "lora"
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)
        logger.info(f"LoRA injected (rank={rank}, alpha={alpha}, targets={target_modules})")

    def inject_vpt(self, token_len: int, init_method: str) -> None:
        self._tuning_mode = "vpt"
        self.model.encoder.backbone.init_prompt_tokens(
            token_len=token_len,
            init_method=init_method,
        )
        logger.info(f"VPT tokens initialized (len={token_len}, method={init_method})")

    def get_trainable_params(self, lr: float) -> list[dict]:
        for param in self.model.parameters():
            param.requires_grad = False

        trainable = []
        if self._tuning_mode == "lora":
            for name, param in self.model.named_parameters():
                if "lora_" in name and "encoder.backbone" in name:
                    param.requires_grad = True
                    trainable.append(param)
        elif self._tuning_mode == "vpt":
            for name, param in self.model.named_parameters():
                if "prompt_tokens" in name:
                    param.requires_grad = True
                    trainable.append(param)

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable: {n_trainable:,} / {n_total:,} ({n_trainable / n_total:.4%})")

        return [{"params": trainable, "lr": lr}]
