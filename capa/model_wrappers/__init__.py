# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""Model registry for CAPA."""

from .base import BaseModel


def get_model(model_name: str) -> BaseModel:
    """Factory function to create model wrapper by name."""
    if model_name == "vggt":
        from .vggt_model import VGGTModel

        return VGGTModel()
    elif model_name == "moge":
        from .moge_model import MoGeV2Model

        return MoGeV2Model()
    elif model_name == "unidepth":
        from .unidepth_model import UniDepthV2Model

        return UniDepthV2Model()
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: vggt, moge, unidepth")
