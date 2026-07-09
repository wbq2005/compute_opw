# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""Abstract base class for CAPA model wrappers."""

import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(ABC):
    """
    Wraps a depth estimation model for use with CAPA protocol.

    Each model wrapper handles:
    - Loading the pretrained model
    - Running depth prediction (forward pass)
    - Injecting LoRA or VPT adaptation parameters
    - Providing trainable parameters for the optimizer
    - Resetting adaptation for the next scene
    """

    model: nn.Module
    device: torch.device
    _model_cpu: nn.Module  # CPU snapshot of original weights for fast reset

    def _cache_original_model(self) -> None:
        """Deepcopy the base model to CPU after load() for fast, guaranteed-clean reset."""
        self.model.cpu()
        self._model_cpu = copy.deepcopy(self.model)
        self.model.to(self.device)

    def _restore_from_cache(self) -> nn.Module:
        """Return a fresh copy of the original model on self.device."""
        return copy.deepcopy(self._model_cpu).to(self.device)

    @abstractmethod
    def load(self, ckpt_path: str, device: torch.device) -> None:
        """Load pretrained model from checkpoint."""
        ...

    @abstractmethod
    def predict_depth(self, rgb_b3hw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run depth prediction.

        Args:
            rgb_b3hw: RGB images [B, 3, H, W] in [0, 1] range

        Returns:
            depth_bhw: predicted depth [B, H, W]
        """
        ...

    @abstractmethod
    def inject_lora(self, rank: int, alpha: int, target_modules: list[str]) -> None:
        """Inject LoRA adaptation layers into the ViT encoder."""
        ...

    @abstractmethod
    def inject_vpt(self, token_len: int, init_method: str) -> None:
        """Inject VPT (Visual Prompt Tuning) tokens into the ViT encoder."""
        ...

    @abstractmethod
    def get_trainable_params(self, lr: float) -> list[dict]:
        """Return parameter groups for the optimizer (only adaptation params)."""
        ...

    def reset(self) -> None:
        """Reset model to pretrained weights (for fresh LoRA/VPT per scene).

        Frees the current GPU model before restoring to avoid a transient 2x GPU peak
        when the old model (possibly LoRA-wrapped) and the new model coexist on GPU.
        """
        self.model = None  # free current GPU model (PeftModel or plain model)
        torch.cuda.empty_cache()
        self.model = self._restore_from_cache()
        self._tuning_mode = None
