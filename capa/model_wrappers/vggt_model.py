# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""VGGT model wrapper for CAPA."""

import sys
import os
import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .base import BaseModel
from ..utils.logging import get_local_logger

logger = get_local_logger(__name__)

# Add base_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "VGGT_VPT"))
from vggt.models.vggt import VGGT  # noqa: E402


class VGGTModel(BaseModel):
    """VGGT depth estimation model with LoRA/VPT support."""

    def __init__(self):
        self.model: nn.Module = None  # VGGT or PeftModel wrapping VGGT
        self.device = None
        self._tuning_mode = None  # "lora" or "vpt"

    @property
    def _base_model(self) -> VGGT:
        """Access the underlying VGGT model (unwrap PeftModel if needed)."""
        if isinstance(self.model, PeftModel):
            return self.model.base_model.model
        return self.model

    def load(self, ckpt_path: str, device: torch.device) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = VGGT.from_pretrained(ckpt_path).eval().to(device)
        self._cache_original_model()
        logger.info(f"VGGT model loaded from {ckpt_path}")

    def predict_depth(self, rgb_b3hw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run VGGT depth prediction.

        Args:
            rgb_b3hw: [B, 3, H, W] in [0, 1], on self.device

        Returns:
            depth_bhw: [B, H, W]
        """
        _, _, height, width = rgb_b3hw.shape
        rgb_processed = self.preprocess_inputs(rgb_b3hw)
        return self.predict_depth_from_processed(
            rgb_processed,
            output_size=(height, width),
        )

    def preprocess_inputs(self, rgb_b3hw: torch.Tensor) -> torch.Tensor:
        """Preprocess VGGT input images once before optimization."""
        return self._process_images(rgb_b3hw.to(self.device))

    def predict_depth_from_processed(
        self,
        rgb_processed_b3hw: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Run VGGT depth prediction from cached preprocessed images.

        Args:
            rgb_processed_b3hw: Preprocessed RGB images [B, 3, h, w].
            output_size: Original output depth size as (height, width).

        Returns:
            depth_bhw: [B, H, W].
        """
        base = self._base_model
        images = rgb_processed_b3hw.unsqueeze(0)

        use_prompt = self._tuning_mode == "vpt"

        # bfloat16 for the ViT aggregator, float32 for the depth head
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            depth_head_feature_idxs = base.depth_head.intermediate_layer_idx
            aggregated_tokens_list, ps_idx = base.aggregator(
                images, use_prompt=use_prompt, feature_idx_ls=depth_head_feature_idxs
            )

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            depth_map, depth_conf = base.depth_head(aggregated_tokens_list, images, ps_idx)

        depth_bhw = depth_map.squeeze(0).squeeze(-1)  # [B, h_out, w_out]
        depth_bhw = resize(
            depth_bhw.unsqueeze(1),
            size=output_size,
            interpolation=InterpolationMode.BILINEAR,
        ).squeeze(1)  # [B, H, W]

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
        self._base_model.aggregator.patch_embed.init_prompt_tokens(
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
                if "lora_" in name and "patch_embed" in name:
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

    @staticmethod
    def _process_images(image_bchw: torch.Tensor) -> torch.Tensor:
        """Resize images for VGGT (518px width, height divisible by 14, center crop if needed)."""
        target_size = 518
        _, _, height, width = image_bchw.shape

        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        image_bchw = resize(
            image_bchw,
            size=(new_height, new_width),
            interpolation=InterpolationMode.BICUBIC,
        )

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            image_bchw = image_bchw[:, :, start_y : start_y + target_size, :]

        return image_bchw
