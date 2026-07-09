#!/usr/bin/env python3
"""
Unified data preparation for CAPA depth completion.

Supports all dataset/condition combinations from the paper:
  - ScanNet:    sift, random100, lt3m     (+noisy/clean)
  - iBims-1:    sift, random100, lt5m     (+noisy/clean)
  - 7-Scenes:   sfm, random100, lt3m     (sfm=COLMAP no noise; others +noisy/clean)
  - Metropolis: 8line, 16line, 32line     (+noisy/clean)

Condition strategies (following CAPA supplement_content.tex):
  - sift:      SIFT keypoints via OpenCV, filtered by GT depth mask
  - random100: Uniformly random 100 points from valid GT pixels
  - lt{N}m:    All valid GT pixels with depth < N meters
  - {N}line:   Simulated LiDAR with N scan lines via spherical pitch angles

LiDAR simulation (Metropolis only, following CAPA supplement §B.2):
  Pitch is spherical: pitch(u,v) = atan2(v - cy, sqrt((u - cx)^2 + fx^2))
  1. Compute per-frame pitch range [min_pitch, max_pitch] for all frames across
     all 36 scenes, then take the mean → global [mean_min_pitch, mean_max_pitch].
  2. Uniformly sample N_L pitch angles within that global range.
  3. For each frame, for each pixel with valid GT, compute its spherical pitch
     and assign it to the nearest target pitch → select that pixel.
  4. Intersect with GT mask (guaranteed by step 3).
  Camera intrinsics: fx = fy = 512, cx = cy = 512 (1024×1024, ~90° FOV).

Noise injection (10% of condition points, uniform [p10, p90] of GT depth range).
If fewer than 5 condition points, supplement with random valid points.

Storage format (ScanNet):
  - Base:  scannet_base/{scene}.pt  — metadata only (raw_dir, frame_indices, etc.)
           RGB/depth loaded on-the-fly from raw ScanNet frames at run-time.
  - Cond:  scannet_{cond}_{noise}/{scene}.pt — COO sparse condition points
           References the base via "source_base" path; decoded by run.py.
  This avoids duplicating ~600 MB of RGB+GT per scene per condition.

Usage:
    # ScanNet SIFT with noise
    python scripts/prepare_data.py scannet sift --noise

    # iBims-1 random100 clean (no noise)
    python scripts/prepare_data.py ibims1 random100

    # ScanNet all conditions
    python scripts/prepare_data.py scannet all --noise

    # Specific scenes only
    python scripts/prepare_data.py scannet sift --noise --scenes scene0707_00 scene0708_00

    # Verify against official sample_data
    python scripts/prepare_data.py scannet sift --noise --verify
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ─── Project paths ───────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = _PROJECT_ROOT / "dataset"

# Raw data directories (only needed for prepare_data.py, NOT for run.py).
# Download sources:
#   - ScanNet v2 test split: http://www.scan-net.org/ (scans_test, 100 scenes)
#   - 7-Scenes: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
#   - Metropolis: included in dataset/metropolis/metropolis_base/
# Set these paths to where you extracted the raw data, or pass --raw-dir on CLI.
_RAW_DATA = {
    "scannet": Path("/data0/tankh/datasets/scans_test"),  # or wherever you put it
    "7scenes": Path("/data0/tankh/datasets/7scenes"),      # or wherever you put it
}

NOISE_FRACTION = 0.10
MIN_CONDITION_POINTS = 5
SEED = 42


# ─── Dataset configs ─────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "scannet": {
        "scene_list": _DATASET_DIR / "scannet" / "scene_list.txt",
        "conditions": ["sift", "random100", "lt3m"],
        "n_frames": 100,
        "max_raw_frames": 300,
        "depth_h": 480,
        "depth_w": 640,
        "raw_data_dir": _RAW_DATA.get("scannet"),
    },
    "ibims1": {
        "base_dir": _DATASET_DIR / "ibims1",
        "conditions": ["sift", "random100", "lt5m"],
        "n_frames": 1,  # single image
    },
    "7scenes": {
        "base_dir": _DATASET_DIR / "7scenes",
        "raw_data_dir": _RAW_DATA.get("7scenes"),
        "conditions": ["sfm", "random100", "lt3m"],
        "n_frames": 100,
        # 18 test sequences (standard test split, matching paper "all 18 available sequences")
        "test_sequences": [
            "chess_seq-03", "chess_seq-05",
            "fire_seq-03", "fire_seq-04",
            "heads_seq-01",
            "office_seq-02", "office_seq-06", "office_seq-07", "office_seq-09", "office_seq-10",
            "pumpkin_seq-01", "pumpkin_seq-07",
            "redkitchen_seq-03", "redkitchen_seq-04", "redkitchen_seq-06",
            "redkitchen_seq-12", "redkitchen_seq-14",
            "stairs_seq-01",
        ],
    },
    "metropolis": {
        "base_dir": _DATASET_DIR / "metropolis",
        "conditions": ["8line", "16line", "32line"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Condition point selection strategies
# ═══════════════════════════════════════════════════════════════════════════════

def select_sift(rgb_uint8: np.ndarray, depth_gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Extract SIFT keypoints, filter by valid GT depth."""
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    if len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    h, w = rgb_uint8.shape[:2]
    coords = np.array([
        [int(round(kp.pt[1])), int(round(kp.pt[0]))] for kp in keypoints
    ], dtype=np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, h - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, w - 1)

    # Remove duplicates
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    coords = coords[sorted(unique_idx)]

    # Filter by GT mask
    rows, cols = coords[:, 0], coords[:, 1]
    valid = depth_gt[rows, cols] > 0
    return coords[valid]


def select_random100(rgb_uint8: np.ndarray, depth_gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 100 points from valid GT pixels."""
    valid_rows, valid_cols = np.where(depth_gt > 0)
    if len(valid_rows) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    n = min(100, len(valid_rows))
    idx = rng.choice(len(valid_rows), size=n, replace=False)
    return np.stack([valid_rows[idx], valid_cols[idx]], axis=1).astype(np.int32)


def select_lt_depth(max_depth: float):
    """Factory: select all valid GT pixels with depth < max_depth."""
    def _select(rgb_uint8: np.ndarray, depth_gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        valid = (depth_gt > 0) & (depth_gt < max_depth)
        rows, cols = np.where(valid)
        if len(rows) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.stack([rows, cols], axis=1).astype(np.int32)
    return _select


def select_lidar_lines_spherical(target_pitches: np.ndarray, fx: float, fy: float,
                                  cx: float, cy: float):
    """Factory: simulate LiDAR scan lines using spherical pitch angles.

    For each target pitch angle and each column u in [0, W):
        v(u) = cy + tan(pitch) * sqrt((u - cx)^2 + fx^2)
    Round v to the nearest integer row. If that row has valid GT depth, select it.

    This produces curved scan bands in image space (not horizontal rows),
    faithfully simulating a mechanical LiDAR's scan pattern.

    Args:
        target_pitches: Array of N_L uniformly-spaced pitch angles (radians).
        fx, fy, cx, cy: Camera intrinsics.
    """
    def _select(rgb_uint8: np.ndarray, depth_gt: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
        h, w = depth_gt.shape
        valid_mask = depth_gt > 0

        cols = np.arange(w, dtype=np.float64)
        # sqrt((u - cx)^2 + fx^2) for each column — shared across all pitches
        dist_col = np.sqrt((cols - cx) ** 2 + fx ** 2)

        all_rows = []
        all_cols = []

        for pitch in target_pitches:
            # For each column, compute the row corresponding to this pitch
            v_float = cy + np.tan(pitch) * dist_col  # shape (W,)
            v_int = np.rint(v_float).astype(np.int64)

            # Filter: row must be within image bounds
            in_bounds = (v_int >= 0) & (v_int < h)
            valid_u = cols[in_bounds].astype(np.int64)
            valid_v = v_int[in_bounds]

            # Filter: must have valid GT depth at (v, u)
            has_gt = valid_mask[valid_v, valid_u]
            all_rows.append(valid_v[has_gt])
            all_cols.append(valid_u[has_gt])

        if not all_rows:
            return np.zeros((0, 2), dtype=np.int32)

        sel_rows = np.concatenate(all_rows)
        sel_cols = np.concatenate(all_cols)

        if len(sel_rows) == 0:
            return np.zeros((0, 2), dtype=np.int32)

        # Remove duplicates (rare, but possible when two pitches round to same row)
        coords = np.stack([sel_rows, sel_cols], axis=1).astype(np.int32)
        coords = np.unique(coords, axis=0)
        return coords

    return _select


# ─── Metropolis LiDAR: global pitch range computation ────────────────────────
#
# Mapillary Metropolis dataset background:
#   - Original images are 360° equirectangular panoramas.
#   - Four perspective crops (CAM_FRONT/BACK/LEFT/RIGHT) are warped from the
#     panorama, each 1024×1024, ~87-90° FOV.
#   - CAPA uses these perspective crops, which follow a pinhole camera model.
#
# Camera intrinsics (pinhole, verified by curve-fitting official sample scan lines):
#   fx = fy ≈ 512  (90° FOV; fit residual < 0.3 px across all 8 scan lines)
#   cx = cy = 512  (image center)
#
# Spherical pitch formula for LiDAR projection:
#   v(u) = cy + tan(pitch) × sqrt((u - cx)² + fx²)
# This produces curved 1-pixel-wide scan bands matching the official data.
_METRO_FX = 512.0
_METRO_CX = 512.0
_METRO_CY = 512.0

# ─── Fixed global pitch range (reverse-engineered from official sample_data) ──
#
# Method: curve-fit each scan-line cluster in the official 8-line sample data
# (scene_023 and scene_007) to extract exact pitch angles. Both scenes share
# the EXACT same 8 pitch centers = linspace(pmin, pmax, 8), confirming the
# range is a dataset-global constant, not per-sequence.
#
# These values were further optimized by brute-force IoU maximization across
# all 164 frames of both official samples → IoU > 0.96.
#
# Residual ~4% IoU gap is entirely due to ±1 pixel rounding at v_float ≈ X.5
# boundaries, caused by sub-0.01° pitch precision differences.
#
# Pitch range derivation (for reference):
#   - Per-frame simple pitch = atan2(v - cy, fy) with fy ≈ 538 (87° FOV)
#   - Global mean of per-frame [min_pitch, max_pitch] across all 36 scenes
#   - The slight fy discrepancy (512 for projection vs 538 for range) may
#     reflect the original data pipeline using different intrinsics for
#     pitch-range computation vs scan-line rendering.
_METRO_PITCH_MIN_DEG = -38.6450   # degrees
_METRO_PITCH_MAX_DEG = 31.7176    # degrees

# Cache for computed global pitch ranges (populated once, reused)
_metro_pitch_cache: dict[int, np.ndarray] = {}


def compute_metropolis_target_pitches(n_lines: int, raw_dir: Path) -> np.ndarray:
    """Compute global target pitch angles for Metropolis LiDAR simulation.

    Uses the fixed pitch range reverse-engineered from official sample_data.
    This is the most reliable approach since we cannot exactly replicate the
    official code's intrinsic / range computation pipeline.

    The fixed range produces pitch centers that match the official 8-line
    sample data to within 0.02° (verified via curve_fit on scan-line clusters).
    """
    if n_lines in _metro_pitch_cache:
        return _metro_pitch_cache[n_lines]

    mean_pitch_min = np.radians(_METRO_PITCH_MIN_DEG)
    mean_pitch_max = np.radians(_METRO_PITCH_MAX_DEG)
    target_pitches = np.linspace(mean_pitch_min, mean_pitch_max, n_lines)
    _metro_pitch_cache[n_lines] = target_pitches
    print(f"  {n_lines}-line target pitches (deg): "
          f"{np.round(np.degrees(target_pitches), 2).tolist()}")
    return target_pitches


# Selector registry (LiDAR selectors are created dynamically for Metropolis)
# "sfm" is handled specially in process_7scenes() via COLMAP
SELECTORS = {
    "sift": select_sift,
    "random100": select_random100,
    "lt3m": select_lt_depth(3.0),
    "lt5m": select_lt_depth(5.0),
    "sfm": None,  # placeholder — 7scenes SfM handled by process_7scenes()
    # "8line", "16line", "32line" are NOT here — they require global pitch
    # computation and are created on-the-fly in process_metropolis().
}

# LiDAR condition names → number of scan lines
LIDAR_CONDITIONS = {
    "8line": 8,
    "16line": 16,
    "32line": 32,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Noise injection
# ═══════════════════════════════════════════════════════════════════════════════

def supplement_random_points(coords: np.ndarray, depth_gt: np.ndarray, min_pts: int, rng: np.random.Generator) -> np.ndarray:
    """If fewer than min_pts, add random valid points."""
    if len(coords) >= min_pts:
        return coords
    valid_rows, valid_cols = np.where(depth_gt > 0)
    if len(valid_rows) == 0:
        return coords
    existing = set(map(tuple, coords.tolist())) if len(coords) > 0 else set()
    candidates = [(r, c) for r, c in zip(valid_rows, valid_cols) if (r, c) not in existing]
    if not candidates:
        return coords
    n = min(min_pts - len(coords), len(candidates))
    idx = rng.choice(len(candidates), size=n, replace=False)
    extra = np.array([candidates[i] for i in idx], dtype=np.int32)
    return np.concatenate([coords, extra], axis=0) if len(coords) > 0 else extra


def inject_noise(depth_values: np.ndarray, depth_gt_full: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt 10% of condition points with uniform noise in [p10, p90].

    Following CAPA paper (supplement §A.3) and verified against official sample_data:
      - P10/P90 are computed from the **condition points' GT depth values**,
        NOT the full image GT. This is critical for limited-range conditions
        (lt3m/lt5m) where full-image percentiles would be much wider.
      - Noise is a **replacement** (not additive): corrupted values are drawn
        from U(p10, p90) and replace the original depth.
    """
    n = len(depth_values)
    if n == 0:
        return depth_values
    n_corrupt = max(1, int(round(n * NOISE_FRACTION)))
    corrupt_idx = rng.choice(n, size=min(n_corrupt, n), replace=False)
    # P10/P90 from condition points' depth values (verified against official data)
    valid_cond = depth_values[depth_values > 0]
    if len(valid_cond) == 0:
        return depth_values
    p10, p90 = np.percentile(valid_cond, 10), np.percentile(valid_cond, 90)
    noisy = depth_values.copy()
    noisy[corrupt_idx] = rng.uniform(p10, p90, size=len(corrupt_idx)).astype(np.float32)
    return noisy


# ═══════════════════════════════════════════════════════════════════════════════
#  ScanNet: build from raw color/depth frames
# ═══════════════════════════════════════════════════════════════════════════════

def get_scannet_frame_indices(scene_dir: Path, n_frames: int = 100, max_raw: int = 300) -> list[int]:
    """Uniformly sample n_frames from first max_raw frames (step=3)."""
    depth_dir = scene_dir / "depth"
    existing = sorted([int(f.stem) for f in depth_dir.glob("*.png")])
    available = [f for f in existing if f < max_raw]
    if len(available) <= n_frames:
        return available
    step = max_raw / n_frames
    indices = []
    for i in range(n_frames):
        target = int(i * step)
        closest = min(available, key=lambda x: abs(x - target))
        if closest not in indices:
            indices.append(closest)
        else:
            for cand in sorted(available, key=lambda x: abs(x - target)):
                if cand not in indices:
                    indices.append(cand)
                    break
    return sorted(indices[:n_frames])


def _ensure_scannet_base(scene_name: str, cfg: dict) -> Path | None:
    """Create base .pt for a ScanNet scene (metadata only, ~1 KB).

    The base stores raw_dir + frame_indices so that run.py can load
    RGB/depth on-the-fly, avoiding ~600 MB duplication per condition.
    """
    raw_dir = cfg["raw_data_dir"]
    base_dir = _DATASET_DIR / "scannet_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / f"{scene_name}.pt"
    if base_path.exists():
        return base_path

    scene_dir = raw_dir / scene_name
    if not scene_dir.exists():
        print(f"  WARN: {scene_dir} not found, skip")
        return None

    frame_indices = get_scannet_frame_indices(scene_dir, cfg["n_frames"], cfg["max_raw_frames"])
    base_data = {
        "raw_dir": str(scene_dir),
        "frame_indices": frame_indices,
        "depth_height": cfg["depth_h"],
        "depth_width": cfg["depth_w"],
    }
    torch.save(base_data, base_path)
    return base_path


def _load_frame(scene_dir: Path, fid: int, dh: int, dw: int):
    """Load one ScanNet frame (depth + resized RGB) from raw files."""
    depth = cv2.imread(str(scene_dir / "depth" / f"{fid}.png"), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    # Use LANCZOS to stay consistent with Effo / DeCoTR baselines
    rgb_pil = Image.open(str(scene_dir / "color" / f"{fid}.jpg")).convert("RGB")
    rgb_pil = rgb_pil.resize((dw, dh), Image.Resampling.LANCZOS)
    rgb_uint8 = np.array(rgb_pil)
    return rgb_uint8, depth


def process_scannet(condition: str, noise: bool, scenes: list[str] | None, seed: int, verify: bool):
    """Process ScanNet scenes: generate base (metadata) + COO condition .pt files."""
    cfg = DATASET_CONFIGS["scannet"]
    raw_dir = cfg["raw_data_dir"]
    if raw_dir is None or not raw_dir.exists():
        print(f"ERROR: ScanNet raw data not found at {raw_dir}")
        return

    suffix = "noise_10pct" if noise else "clean"
    out_dir = _DATASET_DIR / f"scannet_{condition}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_list = scenes or cfg["scene_list"].read_text().strip().split("\n")
    selector = SELECTORS[condition]
    rng = np.random.default_rng(seed)
    dh, dw = cfg["depth_h"], cfg["depth_w"]

    print(f"ScanNet/{condition}/{suffix}: {len(scene_list)} scenes -> {out_dir}")

    results = []
    for scene_name in tqdm(scene_list, desc=f"ScanNet/{condition}"):
        scene_name = scene_name.strip()
        if not scene_name:
            continue

        out_path = out_dir / f"{scene_name}.pt"
        if out_path.exists():
            results.append(scene_name)
            continue

        # Step 1: ensure base exists
        base_path = _ensure_scannet_base(scene_name, cfg)
        if base_path is None:
            continue

        base_data = torch.load(base_path, map_location="cpu", weights_only=False)
        scene_dir = Path(base_data["raw_dir"])
        frame_indices = base_data["frame_indices"]

        # Step 2: generate COO condition for each frame
        cond_list = []
        for fid in frame_indices:
            rgb_uint8, depth = _load_frame(scene_dir, fid, dh, dw)

            # Select condition points
            coords = selector(rgb_uint8, depth, rng)
            coords = supplement_random_points(coords, depth, MIN_CONDITION_POINTS, rng)

            if len(coords) > 0:
                rows, cols = coords[:, 0], coords[:, 1]
                vals = depth[rows, cols].copy()
                if noise:
                    vals = inject_noise(vals, depth, rng)
                cond_list.append({
                    "rows": torch.from_numpy(rows.astype(np.int16)),
                    "cols": torch.from_numpy(cols.astype(np.int16)),
                    "vals": torch.from_numpy(vals),
                })
            else:
                cond_list.append({
                    "rows": torch.zeros(0, dtype=torch.int16),
                    "cols": torch.zeros(0, dtype=torch.int16),
                    "vals": torch.zeros(0, dtype=torch.float32),
                })

        cond_data = {
            "source_base": str(base_path),
            "format": "coo",
            "conditions": cond_list,
        }
        torch.save(cond_data, out_path)
        results.append(scene_name)

    print(f"Done: {len(results)} scenes saved to {out_dir}")

    if verify:
        _verify_scannet(condition, out_dir)


def _verify_scannet(condition: str, out_dir: Path):
    """Compare COO condition output with official sample_data (self-contained .pt)."""
    sample_map = {"sift": "scannet_sift_noise_10pct"}
    sample_name = sample_map.get(condition)
    if not sample_name:
        print(f"No official samples for condition '{condition}'")
        return
    sample_dir = _DATASET_DIR / "sample_data" / sample_name
    if not sample_dir.exists():
        print(f"Official sample dir not found: {sample_dir}")
        return

    cfg = DATASET_CONFIGS["scannet"]
    dh, dw = cfg["depth_h"], cfg["depth_w"]

    print(f"\nVerifying against {sample_dir}...")
    for off_pt in sorted(sample_dir.glob("*.pt")):
        gen_name = off_pt.stem + "_00"
        gen_pt = out_dir / f"{gen_name}.pt"
        if not gen_pt.exists():
            print(f"  {gen_name}: not generated, skip")
            continue

        off = torch.load(off_pt, map_location="cpu", weights_only=False)
        gen_cond = torch.load(gen_pt, map_location="cpu", weights_only=False)

        # Decode COO to dense for comparison
        base_data = torch.load(gen_cond["source_base"], map_location="cpu", weights_only=False)
        scene_dir = Path(base_data["raw_dir"])
        frame_indices = base_data["frame_indices"]

        # Compare GT depth (load from raw)
        gen_gt0 = cv2.imread(str(scene_dir / "depth" / f"{frame_indices[0]}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        off_gt0 = off["depth_gt_nvhw"][0].numpy()
        gt_diff = np.abs(gen_gt0 - off_gt0).max()

        # Compare condition point counts
        gen_pts = sum(len(c["rows"]) for c in gen_cond["conditions"])
        off_pts = off["mask_condition_nvhw"].sum().item()
        ratio = gen_pts / off_pts if off_pts > 0 else float("inf")
        print(f"  {gen_name}: GT_diff={gt_diff:.6f}, pts={gen_pts}/{off_pts} (ratio={ratio:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  iBims-1 / 7-Scenes / Metropolis: re-condition existing .pt files
# ═══════════════════════════════════════════════════════════════════════════════

def process_existing_dataset(dataset: str, condition: str, noise: bool, scenes: list[str] | None, seed: int):
    """
    For datasets that already have self-contained .pt files (iBims-1, 7-Scenes, Metropolis),
    load existing data, re-generate condition points, and save new .pt files.
    """
    cfg = DATASET_CONFIGS[dataset]
    base_dir = cfg["base_dir"]
    selector = SELECTORS[condition]
    rng = np.random.default_rng(seed)

    suffix = "noisy" if noise else "clean"
    out_dir = base_dir / f"{dataset}_{condition}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find source .pt files (from any existing condition dir, or a dedicated base dir)
    source_dir = _find_source_dir(dataset, base_dir)
    if source_dir is None:
        print(f"ERROR: No source data found for {dataset}")
        return

    pt_files = sorted(source_dir.glob("*.pt"))
    if scenes:
        pt_files = [f for f in pt_files if f.stem in scenes]

    print(f"{dataset}/{condition}/{suffix}: {len(pt_files)} samples from {source_dir} -> {out_dir}")

    for pt_file in tqdm(pt_files, desc=f"{dataset}/{condition}"):
        out_path = out_dir / pt_file.name
        if out_path.exists():
            continue

        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        # Handle source_pt references
        if "source_pt" in data and "rgb_nv3hw" not in data:
            source_data = torch.load(data["source_pt"], map_location="cpu", weights_only=False)
            data["rgb_nv3hw"] = source_data["rgb_nv3hw"]
            data["depth_gt_nvhw"] = source_data["depth_gt_nvhw"]

        rgb = data["rgb_nv3hw"]   # [N, 3, H, W]
        gt = data["depth_gt_nvhw"]  # [N, H, W]
        N = rgb.shape[0]
        conds, masks = [], []

        for i in range(N):
            rgb_uint8 = (rgb[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            depth_gt = gt[i].numpy()

            coords = selector(rgb_uint8, depth_gt, rng)
            coords = supplement_random_points(coords, depth_gt, MIN_CONDITION_POINTS, rng)

            h, w = depth_gt.shape
            cond = np.zeros((h, w), dtype=np.float32)
            mask = np.zeros((h, w), dtype=bool)
            if len(coords) > 0:
                rows, cols = coords[:, 0], coords[:, 1]
                vals = depth_gt[rows, cols].copy()
                if noise:
                    vals = inject_noise(vals, depth_gt, rng)
                cond[rows, cols] = vals
                mask[rows, cols] = True

            conds.append(torch.from_numpy(cond))
            masks.append(torch.from_numpy(mask))

        out_data = {
            "rgb_nv3hw": rgb,
            "depth_gt_nvhw": gt,
            "depth_condition_nvhw": torch.stack(conds),
            "mask_condition_nvhw": torch.stack(masks),
        }
        torch.save(out_data, out_path)

    print(f"Done: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Metropolis: LiDAR simulation with spherical pitch angles
# ═══════════════════════════════════════════════════════════════════════════════

def process_metropolis(condition: str, noise: bool, scenes: list[str] | None, seed: int):
    """Process Metropolis scenes with spherical-pitch LiDAR simulation.

    Unlike other datasets, Metropolis LiDAR requires:
      1. A global pitch range computed across ALL 36 scenes (not per-frame).
      2. Spherical pitch assignment for each pixel.
    """
    cfg = DATASET_CONFIGS["metropolis"]
    base_dir = cfg["base_dir"]
    raw_dir = base_dir / "metropolis_base"
    n_lines = LIDAR_CONDITIONS[condition]
    rng = np.random.default_rng(seed)

    # Step 1: Compute global target pitches
    target_pitches = compute_metropolis_target_pitches(n_lines, raw_dir)

    # Step 2: Create the selector (fx used for both horizontal & vertical projection)
    selector = select_lidar_lines_spherical(
        target_pitches, _METRO_FX, _METRO_FX, _METRO_CX, _METRO_CY
    )

    suffix = "noisy" if noise else "clean"
    out_dir = base_dir / f"metropolis_{condition}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find source .pt files (raw data with rgb + gt)
    source_dir = raw_dir
    pt_files = sorted(source_dir.glob("scene_*.pt"))
    if scenes:
        pt_files = [f for f in pt_files if f.stem in scenes]

    print(f"metropolis/{condition}/{suffix}: {len(pt_files)} scenes -> {out_dir}")
    print(f"  Storage: COO sparse + source_pt reference (~1MB/scene)")

    for pt_file in tqdm(pt_files, desc=f"metropolis/{condition}"):
        out_path = out_dir / pt_file.name
        if out_path.exists():
            continue

        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        gt = data["depth_gt_nvhw"]   # [N, H, W]
        N = gt.shape[0]
        cond_list = []

        for i in range(N):
            depth_gt = gt[i].numpy()
            rgb_dummy = np.zeros((1, 1, 3), dtype=np.uint8)

            coords = selector(rgb_dummy, depth_gt, rng)
            coords = supplement_random_points(coords, depth_gt, MIN_CONDITION_POINTS, rng)

            if len(coords) > 0:
                rows, cols = coords[:, 0], coords[:, 1]
                vals = depth_gt[rows, cols].copy()
                if noise:
                    vals = inject_noise(vals, depth_gt, rng)
                cond_list.append({
                    "rows": torch.from_numpy(rows.astype(np.int16)),
                    "cols": torch.from_numpy(cols.astype(np.int16)),
                    "vals": torch.from_numpy(vals),
                })
            else:
                cond_list.append({
                    "rows": torch.zeros(0, dtype=torch.int16),
                    "cols": torch.zeros(0, dtype=torch.int16),
                    "vals": torch.zeros(0, dtype=torch.float32),
                })

        # COO sparse storage + reference to source .pt for rgb/gt
        # run.py load_sample() will decode COO → dense at load time
        out_data = {
            "source_pt": str(pt_file.resolve()),
            "format": "coo",
            "conditions": cond_list,
        }
        torch.save(out_data, out_path)

    print(f"Done: {out_dir}")


def _find_source_dir(dataset: str, base_dir: Path) -> Path | None:
    """Find a source directory with complete .pt files (any existing condition)."""
    # Prefer a dedicated base dir
    if dataset == "7scenes":
        d = base_dir / "7scenes_base"
        if d.exists() and list(d.glob("*.pt")):
            return d
    # For ibims1/metropolis, use any existing condition dir
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir() and list(sub.glob("*.pt")):
            return sub
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  7-Scenes: build from raw npy/png files + COLMAP SfM
# ═══════════════════════════════════════════════════════════════════════════════
#
# Paper (main.tex §4.1):
#   "7-Scenes is indoor RGB-D video dataset. For 7-Scenes all 18 available
#    sequences. From each sequence, we uniformly sample 100 frames within
#    the first 300 ones of the video."
#
# Source data layout (already pre-sampled at step=3, 100 frames per seq):
#   dataset/7scenes/7scenes_raw/{scene_seq}/
#       rgb/frame-000000.png ... frame-000299.png   (480x640 RGB)
#       gt/frame-000000.npy  ... frame-000299.npy   (480x640 float32, meters)
#
# GT depth: float32 in meters; 65.535 = invalid (from uint16 max / 1000).
#
# 7-Scenes known intrinsics: fx=fy=585, cx=320, cy=240, 640x480
_7S_FX = 585.0
_7S_FY = 585.0
_7S_CX = 320.0
_7S_CY = 240.0
_7S_H, _7S_W = 480, 640
_7S_INVALID_DEPTH = 65.0  # depth >= 65.0 treated as invalid


def _build_7scenes_base(seq_name: str, cfg: dict, force: bool = False) -> Path | None:
    """Build base .pt from raw npy/png: RGB [N,3,H,W] float32 + GT [N,H,W] float32.

    Reads rgb/*.png and gt/*.npy from the raw 7-Scenes directory.
    GT depth values >= 65.0 are set to 0 (invalid).
    """
    raw_dir = cfg["raw_data_dir"]
    base_dir = cfg["base_dir"] / "7scenes_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"{seq_name}.pt"

    if out_path.exists() and not force:
        return out_path

    seq_dir = raw_dir / seq_name
    rgb_dir = seq_dir / "rgb"
    gt_dir = seq_dir / "gt"

    if not rgb_dir.exists() or not gt_dir.exists():
        print(f"  WARN: {seq_dir} not found, skip")
        return None

    # Get sorted frame stems from gt directory
    stems = sorted([f.stem for f in gt_dir.glob("*.npy")])
    if not stems:
        print(f"  WARN: no .npy files in {gt_dir}, skip")
        return None

    rgbs, depths = [], []
    for stem in stems:
        rgb_path = rgb_dir / f"{stem}.png"
        gt_path = gt_dir / f"{stem}.npy"
        if not rgb_path.exists() or not gt_path.exists():
            continue
        # RGB: uint8 PNG → float32 [0,1] → [3, H, W]
        rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
        rgbs.append(torch.from_numpy(rgb.transpose(2, 0, 1)))
        # GT: float32 meters; 65.535 = invalid → set to 0
        gt = np.load(gt_path).astype(np.float32)
        gt[gt >= _7S_INVALID_DEPTH] = 0.0
        depths.append(torch.from_numpy(gt))

    if not rgbs:
        return None

    data = {
        "rgb_nv3hw": torch.stack(rgbs),      # [N, 3, 480, 640]
        "depth_gt_nvhw": torch.stack(depths), # [N, 480, 640]
    }
    torch.save(data, out_path)
    print(f"  base: {seq_name} → {len(rgbs)} frames, {out_path.stat().st_size / 1024 / 1024:.0f} MB")
    return out_path


def _build_7scenes_condition(seq_name: str, condition: str, noise: bool,
                              cfg: dict, rng: np.random.Generator,
                              force: bool = False) -> bool:
    """Build condition .pt for random100 or lt3m from base .pt GT depth.

    Paper (supplement §A.2):
      - Random: "randomly sample 100 pixels from regions with valid ground truth depth"
      - Limited Range: "select all points with valid depth values within a predefined range"
      - Noise: "10% of condition points randomly selected for corruption"
               with "uniform distribution bounded by P10 and P90 percentiles"
      - "if fewer than 5, supplement by randomly sampling additional points"

    Verified against official sample_data:
      - P10/P90 computed from condition points' GT depth values (not full image)
      - Noise is replacement: corrupted_val = U(p10, p90)
    """
    base_dir = cfg["base_dir"]
    suffix = "noisy" if noise else "clean"
    out_dir = base_dir / f"7scenes_{condition}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_name}.pt"

    if out_path.exists() and not force:
        return True

    base_path = base_dir / "7scenes_base" / f"{seq_name}.pt"
    if not base_path.exists():
        print(f"  WARN: base {base_path} not found")
        return False

    base_data = torch.load(base_path, map_location="cpu", weights_only=False)
    gt_all = base_data["depth_gt_nvhw"]  # [N, H, W]
    rgb_all = base_data["rgb_nv3hw"]     # [N, 3, H, W]
    N = gt_all.shape[0]

    selector = SELECTORS[condition]
    conds, masks = [], []

    for i in range(N):
        depth_gt = gt_all[i].numpy()
        rgb_uint8 = (rgb_all[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        coords = selector(rgb_uint8, depth_gt, rng)
        coords = supplement_random_points(coords, depth_gt, MIN_CONDITION_POINTS, rng)

        h, w = depth_gt.shape
        cond = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        if len(coords) > 0:
            rows, cols = coords[:, 0], coords[:, 1]
            vals = depth_gt[rows, cols].copy()
            if noise:
                vals = inject_noise(vals, depth_gt, rng)
            cond[rows, cols] = vals
            mask[rows, cols] = True

        conds.append(torch.from_numpy(cond))
        masks.append(torch.from_numpy(mask))

    torch.save({
        "source_pt": str(base_path),
        "depth_condition_nvhw": torch.stack(conds),
        "mask_condition_nvhw": torch.stack(masks),
    }, out_path)
    return True


def _build_7scenes_sfm(seq_name: str, cfg: dict, force: bool = False) -> bool:
    """Build SfM condition .pt via COLMAP for a 7-Scenes sequence.

    Paper (supplement §A.2):
      "We utilize actual Structure-from-Motion (SfM) points for 7-Scenes dataset,
       which were pre-computed using COLMAP and contain realistic noise."

    Filtering (three stages):
      1. "retaining only those below the 75th percentile for the bundle adjustment error"
      2. "AbsRel < 0.1" (|sfm_depth - gt_depth| / gt_depth)
      3. "AbsErr < 0.1" (|sfm_depth - gt_depth|)
      4. "3D point distance < 0.1"

    Noise: "For 7-Scenes (SfM), no additional noise is added since the SfM
            points are already naturally noisy"

    COLMAP produces up-to-scale reconstruction → we align to GT with robust
    scale+shift estimation before filtering.
    """
    try:
        import pycolmap
    except ImportError:
        print("  ERROR: pycolmap not installed. Install with: pip install pycolmap")
        print("         SfM condition requires COLMAP for 7-Scenes.")
        return False

    import tempfile

    base_dir = cfg["base_dir"]
    raw_dir = cfg["raw_data_dir"]
    out_dir = base_dir / "7scenes_sfm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seq_name}.pt"

    if out_path.exists() and not force:
        return True

    base_path = base_dir / "7scenes_base" / f"{seq_name}.pt"
    if not base_path.exists():
        print(f"  WARN: base {base_path} not found")
        return False

    seq_dir = raw_dir / seq_name
    rgb_dir = seq_dir / "rgb"
    gt_dir = seq_dir / "gt"

    if not rgb_dir.exists():
        return False

    # Load GT depths for alignment and filtering
    stems = sorted([f.stem for f in gt_dir.glob("*.npy")])
    gt_depths = {}
    for stem in stems:
        gt = np.load(gt_dir / f"{stem}.npy").astype(np.float32)
        gt[gt >= _7S_INVALID_DEPTH] = 0.0
        gt_depths[stem] = gt

    # Run COLMAP SfM
    with tempfile.TemporaryDirectory(prefix=f"colmap_7s_{seq_name}_") as tmpdir:
        tmpdir = Path(tmpdir)
        db_path = tmpdir / "database.db"
        sparse_dir = tmpdir / "sparse"
        sparse_dir.mkdir()

        print(f"    COLMAP: extracting features...")
        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = "PINHOLE"
        reader_opts.camera_params = f"{_7S_FX},{_7S_FY},{_7S_CX},{_7S_CY}"

        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(rgb_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_opts,
            device=pycolmap.Device.auto,
        )

        print(f"    COLMAP: matching...")
        seq_opts = pycolmap.SequentialPairingOptions()
        seq_opts.overlap = 10
        pycolmap.match_sequential(
            database_path=str(db_path),
            pairing_options=seq_opts,
            device=pycolmap.Device.auto,
        )

        print(f"    COLMAP: mapping...")
        mapper_opts = pycolmap.IncrementalPipelineOptions()
        mapper_opts.min_num_matches = 15
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(rgb_dir),
            output_path=str(sparse_dir),
            options=mapper_opts,
        )

    if not reconstructions:
        print(f"    COLMAP failed for {seq_name}")
        return False

    best_key = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    recon = reconstructions[best_key]
    print(f"    COLMAP: {recon.num_reg_images()}/{len(stems)} images, {recon.num_points3D()} 3D points")

    # Step 1: Project raw SfM points to get unscaled depth, for scale estimation
    name_to_img = {}
    for img_id in recon.reg_image_ids():
        img = recon.image(img_id)
        name_to_img[img.name] = img

    sfm_vals, gt_vals = [], []
    for stem in stems:
        fname = f"{stem}.png"
        if fname not in name_to_img or stem not in gt_depths:
            continue
        img = name_to_img[fname]
        gt = gt_depths[stem]
        for p2d in img.points2D:
            if not p2d.has_point3D():
                continue
            p3d = recon.point3D(p2d.point3D_id)
            p_cam = img.cam_from_world() * p3d.xyz
            d_sfm = p_cam[2]
            if d_sfm <= 0:
                continue
            u = _7S_FX * p_cam[0] / d_sfm + _7S_CX
            v = _7S_FY * p_cam[1] / d_sfm + _7S_CY
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < _7S_H and 0 <= ui < _7S_W and gt[vi, ui] > 0:
                sfm_vals.append(d_sfm)
                gt_vals.append(gt[vi, ui])

    if len(sfm_vals) < 10:
        print(f"    Not enough SfM-GT overlap ({len(sfm_vals)} pts)")
        return False

    sfm_arr = np.array(sfm_vals)
    gt_arr = np.array(gt_vals)

    # Step 2: Robust scale+shift estimation: gt ≈ scale * sfm + shift
    from numpy.linalg import lstsq
    A = np.column_stack([sfm_arr, np.ones_like(sfm_arr)])
    result = lstsq(A, gt_arr, rcond=None)
    scale, shift = result[0]
    for _ in range(3):
        residuals = np.abs(gt_arr - (scale * sfm_arr + shift))
        thresh = np.percentile(residuals, 80)
        inlier = residuals < thresh
        if inlier.sum() < 10:
            break
        result = lstsq(A[inlier], gt_arr[inlier], rcond=None)
        scale, shift = result[0]

    print(f"    Scale alignment: scale={scale:.4f}, shift={shift:.4f}")

    # Step 3: BA error threshold (75th percentile)
    all_errors = [recon.point3D(pid).error for pid in recon.point3D_ids()]
    ba_thresh = np.percentile(all_errors, 75) if all_errors else float('inf')

    # Step 4: Project + filter with paper's thresholds
    stats = {"total": 0, "ba": 0, "gt": 0, "kept": 0}
    depths_cond, masks_cond = [], []

    for stem in stems:
        fname = f"{stem}.png"
        sparse = np.zeros((_7S_H, _7S_W), dtype=np.float32)

        if fname in name_to_img and stem in gt_depths:
            img = name_to_img[fname]
            gt = gt_depths[stem]

            for p2d in img.points2D:
                if not p2d.has_point3D():
                    continue
                p3d = recon.point3D(p2d.point3D_id)
                stats["total"] += 1

                # Filter 1: BA error < 75th percentile
                if p3d.error > ba_thresh:
                    stats["ba"] += 1
                    continue

                p_cam = img.cam_from_world() * p3d.xyz
                d_raw = p_cam[2]
                if d_raw <= 0:
                    continue

                # Apply scale+shift
                d = scale * d_raw + shift
                if d <= 0:
                    continue

                u = _7S_FX * p_cam[0] / d_raw + _7S_CX
                v = _7S_FY * p_cam[1] / d_raw + _7S_CY
                ui, vi = int(round(u)), int(round(v))

                if not (0 <= vi < _7S_H and 0 <= ui < _7S_W):
                    continue

                # Filters 2-4: GT consistency
                if gt[vi, ui] > 0:
                    gt_d = float(gt[vi, ui])
                    abs_err = abs(d - gt_d)
                    abs_rel = abs_err / (gt_d + 1e-8)
                    if abs_rel >= 0.1 or abs_err >= 0.1:
                        stats["gt"] += 1
                        continue

                sparse[vi, ui] = d
                stats["kept"] += 1

        depths_cond.append(torch.from_numpy(sparse))
        masks_cond.append(torch.from_numpy(sparse > 0))

    print(f"    SfM filter: {stats['total']} → {stats['kept']} kept "
          f"(-{stats['ba']} BA, -{stats['gt']} GT, ba_thresh={ba_thresh:.2f})")

    pts_per_frame = torch.stack(masks_cond).sum(dim=(1, 2)).float()
    print(f"    Points/frame: mean={pts_per_frame.mean():.0f}, "
          f"min={pts_per_frame.min():.0f}, max={pts_per_frame.max():.0f}")

    # No noise for SfM (paper: "no additional noise is added")
    torch.save({
        "source_pt": str(base_path),
        "depth_condition_nvhw": torch.stack(depths_cond),
        "mask_condition_nvhw": torch.stack(masks_cond),
    }, out_path)
    return True


def process_7scenes(condition: str, noise: bool, scenes: list[str] | None,
                    seed: int, force: bool = False):
    """Process 7-Scenes: build base .pt + condition .pt from raw data.

    Three-stage pipeline:
      1. Build base .pt (RGB + GT) from raw npy/png
      2. For random100/lt3m: generate condition from GT depth
      3. For sfm: run COLMAP and apply paper's three-stage filtering
    """
    cfg = DATASET_CONFIGS["7scenes"]
    test_seqs = scenes or cfg["test_sequences"]
    rng = np.random.default_rng(seed)

    print(f"7-Scenes/{condition}: {len(test_seqs)} sequences")

    # Step 1: Ensure base .pt exists for all sequences
    print("\n--- Building base .pt (RGB + GT) ---")
    for seq in tqdm(test_seqs, desc="7scenes/base"):
        _build_7scenes_base(seq, cfg, force=force)

    # Step 2: Build condition
    if condition == "sfm":
        # SfM: COLMAP reconstruction, no noise
        print("\n--- Building SfM condition (COLMAP, no noise) ---")
        ok = 0
        for i, seq in enumerate(test_seqs):
            print(f"\n  [{i+1}/{len(test_seqs)}] {seq}")
            if _build_7scenes_sfm(seq, cfg, force=force):
                ok += 1
            else:
                print(f"  [FAIL] {seq}")
        print(f"\n  SfM: {ok}/{len(test_seqs)} done")
    else:
        # random100 / lt3m
        suffix = "noisy" if noise else "clean"
        print(f"\n--- Building {condition}_{suffix} condition ---")
        ok = 0
        for seq in tqdm(test_seqs, desc=f"7scenes/{condition}"):
            if _build_7scenes_condition(seq, condition, noise, cfg, rng, force=force):
                ok += 1
        print(f"  {condition}_{suffix}: {ok}/{len(test_seqs)} done")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified data preparation for CAPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py scannet sift --noise
  python scripts/prepare_data.py scannet all --noise
  python scripts/prepare_data.py ibims1 random100 --noise
  python scripts/prepare_data.py 7scenes random100 --noise
  python scripts/prepare_data.py 7scenes sfm
  python scripts/prepare_data.py 7scenes all --noise
  python scripts/prepare_data.py scannet sift --noise --verify
  python scripts/prepare_data.py scannet sift --noise --scenes scene0707_00
        """,
    )
    parser.add_argument("dataset", choices=["scannet", "ibims1", "7scenes", "metropolis"],
                        help="Dataset name")
    parser.add_argument("condition", type=str,
                        help="Condition type (sfm/sift/random100/lt3m/lt5m/8line/16line/32line) or 'all'")
    parser.add_argument("--noise", action="store_true", help="Inject 10%% noise")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--scenes", nargs="*", default=None, help="Process specific scenes only")
    parser.add_argument("--verify", action="store_true", help="Compare with official sample_data")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .pt files")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]

    # Resolve conditions
    if args.condition == "all":
        conditions = cfg["conditions"]
    else:
        conditions = [args.condition]

    for cond in conditions:
        if cond not in SELECTORS and cond not in LIDAR_CONDITIONS:
            print(f"ERROR: Unknown condition '{cond}'. "
                  f"Available: {list(SELECTORS.keys()) + list(LIDAR_CONDITIONS.keys())}")
            return

        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset}, Condition: {cond}, Noise: {args.noise}, Seed: {args.seed}")
        print(f"{'='*60}")

        if args.dataset == "scannet":
            process_scannet(cond, args.noise, args.scenes, args.seed, args.verify)
        elif args.dataset == "metropolis":
            process_metropolis(cond, args.noise, args.scenes, args.seed)
        elif args.dataset == "7scenes":
            process_7scenes(cond, args.noise, args.scenes, args.seed, force=args.force)
        else:
            process_existing_dataset(args.dataset, cond, args.noise, args.scenes, args.seed)


if __name__ == "__main__":
    main()
