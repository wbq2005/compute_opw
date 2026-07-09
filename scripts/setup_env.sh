#!/bin/bash
# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Setup virtual environment(s) for CAPA.
# Each base model ships its own requirements.txt; we install torch first
# (to control the CUDA build), then the model's requirements, then capa.
#
# Usage:
#   bash scripts/setup_env.sh vggt
#   bash scripts/setup_env.sh moge
#   bash scripts/setup_env.sh unidepth
#   bash scripts/setup_env.sh all
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

BASE_MODELS_DIR="$PROJ_ROOT/third_party"
VGGT_DIR="$BASE_MODELS_DIR/VGGT_VPT"
MOGE_DIR="$BASE_MODELS_DIR/MoGe_VPT"
UNIDEPTH_DIR="$BASE_MODELS_DIR/UniDepth_VPT"

setup_env() {
    local name="$1"
    local env_dir="$PROJ_ROOT/venv/${name}_env"

    if [ -d "$env_dir" ]; then
        echo "[$name] Environment already exists: $env_dir"
        echo "[$name] To recreate, delete it first: rm -rf $env_dir"
        return 0
    fi

    echo "[$name] Creating environment..."
    virtualenv "$env_dir" --python=python3
    source "$env_dir/bin/activate"


    # Base model dependencies from their own requirements.txt
    # Filter out torch/torchvision/torchaudio lines to keep our CUDA build
    case "$name" in
        vggt)
            pip install -r "$VGGT_DIR/requirements.txt"
            ;;
        moge)
            pip install -r "$MOGE_DIR/requirements.txt"
            ;;
        unidepth)
            pip install -r "$UNIDEPTH_DIR/requirements.txt"
            ;;
    esac

    # CAPA-level dependencies (peft, matplotlib, etc.)
    pip install -e "$PROJ_ROOT"

    echo "[$name] Environment ready: $env_dir"
    echo "[$name] Activate with: source $env_dir/bin/activate"
}

# --- Main ---
MODELS=("$@")

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: bash scripts/setup_env.sh {vggt|moge|unidepth|all}"
    exit 1
fi

if [ "${MODELS[0]}" = "all" ]; then
    MODELS=(vggt moge unidepth)
fi

for model in "${MODELS[@]}"; do
    case "$model" in
        vggt|moge|unidepth)
            setup_env "$model"
            ;;
        *)
            echo "Unknown model: $model (expected vggt, moge, unidepth, or all)"
            exit 1
            ;;
    esac
done
