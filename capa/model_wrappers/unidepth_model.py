# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""UniDepth-v2 model wrapper for CAPA."""

import json
import sys
import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from .base import BaseModel
from ..utils.logging import get_local_logger

logger = get_local_logger(__name__)

# Add base_models to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "UniDepth_VPT")
)
from unidepth.models import UniDepthV2  # noqa: E402


class UniDepthV2Model(BaseModel):
    """UniDepth-v2 depth estimation model with LoRA/VPT support."""

    def __init__(self):
        self.model: UniDepthV2 = None
        self.device = None
        self._tuning_mode = None

    def _cache_original_model(self) -> None:
        """Save weights as CPU state dict.

        UniDepth contains torch.jit.ScriptFunction attributes that cannot be pickled,
        so copy.deepcopy (used in the base class) fails. We use state_dict instead.
        """
        self._cached_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self._model_init_kwargs = self._load_model_init_kwargs()

    def _load_model_init_kwargs(self) -> dict:
        """Read the init kwargs dict from the checkpoint's config.json."""
        if os.path.isdir(self.ckpt_path):
            config_path = os.path.join(self.ckpt_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=self.ckpt_path, filename="config.json")
        with open(config_path) as f:
            return json.load(f)

    def _restore_from_cache(self) -> nn.Module:
        """Create a fresh UniDepth model from config + cached weights (no disk read for weights)."""
        model = UniDepthV2(config=self._model_init_kwargs).eval()
        model.load_state_dict(self._cached_state_dict)
        return model.to(self.device)

    def load(self, ckpt_path: str, device: torch.device) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = UniDepthV2.from_pretrained(ckpt_path).eval().to(device)
        self._cache_original_model()
        logger.info(f"UniDepth-v2 model loaded from {ckpt_path}")

    def predict_depth(self, rgb_b3hw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run UniDepth-v2 depth prediction.

        Args:
            rgb_b3hw: [B, 3, H, W] in [0, 1], on self.device

        Returns:
            depth_bhw: [B, H, W]
        """
        use_prompt = self._tuning_mode == "vpt"
        # UniDepth expects RGB in [0, 255] range; runs in fp32 (no autocast)
        output = self.model.infer(rgb_b3hw * 255.0, use_prompt=use_prompt)
        depth_bhw = output["depth"].squeeze(1)  # [B, 1, H, W] -> [B, H, W]
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
        self.model.pixel_encoder.init_prompt_tokens(
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
                if "lora_" in name and "pixel_encoder" in name:
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
