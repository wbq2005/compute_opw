#!/usr/bin/env python3
"""Merge an existing offline OPW audit into summary.json.

Use this for an existing audit JSON or for merging disjoint audit shards.
Normal full runs of ``audit_opw_metric.py`` update the adjacent summary
automatically::

    python scripts/merge_opw_summary.py \
      --summary output/noise_probe/metropolis_8line_v3/vggt/summary.json \
      --opw-json output/noise_probe/metropolis_8line_v3/vggt/opw_capa_strict_fb.json

The command validates exact scene coverage, backs up the original summary, and
writes standard JSON atomically. Never copy the reported mean by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

OPW_PROTOCOL = "capa_strict"


def _scene_name(item: dict[str, Any]) -> str | None:
    for key in ("scene", "sample", "name", "stem", "id"):
        value = item.get(key)
        if isinstance(value, str):
            return Path(value).stem.removesuffix("_pred")
    return None


def _summary_scene_items(
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Return the canonical per-scene list and reject ambiguous coverage."""
    for list_key in ("samples", "results", "per_sample", "scene_results"):
        items = payload.get(list_key)
        if not isinstance(items, list):
            continue
        scenes: list[str] = []
        seen: set[str] = set()
        duplicates: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"{list_key}[{index}] must be an object")
            scene = _scene_name(item)
            if scene is None:
                raise ValueError(f"{list_key}[{index}] has no recognizable scene name")
            if scene in seen:
                duplicates.add(scene)
            seen.add(scene)
            scenes.append(scene)
        if duplicates:
            raise ValueError(
                f"Duplicate scenes in summary {list_key}: {sorted(duplicates)}"
            )
        return list_key, items, scenes
    raise ValueError(
        "Summary has no per-scene list; expected one of samples, results, "
        "per_sample, or scene_results"
    )


def _update_per_scene(
    items: list[dict[str, Any]],
    scenes: list[str],
    opw_by_scene: dict[str, float],
) -> int:
    for item, scene in zip(items, scenes):
        metrics = item.get("metrics")
        if isinstance(metrics, dict):
            metrics["opw"] = opw_by_scene[scene]
        else:
            item["opw"] = opw_by_scene[scene]
    return len(items)


def _json_safe(value: Any) -> Any:
    """Convert non-finite legacy floats to JSON-standard null values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _load_strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{path}: non-standard JSON constant {token!r}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _merge_audits(opw_paths: list[Path]) -> tuple[dict[str, Any], dict[str, float]]:
    if not opw_paths:
        raise ValueError("At least one OPW audit JSON is required")

    audits = [_load_strict_json(path) for path in opw_paths]
    for path, audit in zip(opw_paths, audits):
        # Older completed audits used ``opw_mode`` before the strict-only API.
        protocol = audit.get("protocol", audit.pop("opw_mode", None))
        if protocol != OPW_PROTOCOL:
            raise ValueError(
                f"{path}: expected OPW protocol {OPW_PROTOCOL!r}, got {protocol!r}"
            )
        audit["protocol"] = protocol
    shared_keys = (
        "input_dir",
        "pred_dir",
        "gmflow_ckpt",
        "gmflow_repo",
        "beta",
        "fb_consistency",
        "protocol",
        "depth_key",
        "eval_mask_key",
        "flow_batch_size",
        "flow_max_side",
    )
    for key in shared_keys:
        values = [audit.get(key) for audit in audits]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"OPW audit shards disagree on {key!r}: {values!r}")

    opw_by_scene: dict[str, float] = {}
    for path, audit in zip(opw_paths, audits):
        scene_items = audit.get("per_scene_opw")
        if not isinstance(scene_items, list):
            raise ValueError(f"{path}: per_scene_opw must be a list")
        shard_values: list[float] = []
        for item in scene_items:
            if not isinstance(item, dict):
                continue
            scene = item.get("scene")
            opw = item.get("opw")
            if (
                not isinstance(scene, str)
                or isinstance(opw, bool)
                or not isinstance(opw, (int, float))
            ):
                continue
            opw = float(opw)
            if not math.isfinite(opw):
                continue
            if scene in opw_by_scene:
                raise ValueError(f"Duplicate scene {scene!r} across OPW audit shards")
            opw_by_scene[scene] = opw
            shard_values.append(opw)

        if not shard_values:
            raise ValueError(f"{path}: contains no finite per-scene OPW values")
        declared_mean = audit.get("mean_opw")
        shard_mean = sum(shard_values) / len(shard_values)
        if declared_mean is not None and (
            isinstance(declared_mean, bool)
            or not isinstance(declared_mean, (int, float))
            or not math.isfinite(float(declared_mean))
            or not math.isclose(
                float(declared_mean), shard_mean, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(
                f"{path}: mean_opw {declared_mean!r} disagrees with "
                f"per-scene mean {shard_mean!r}"
            )
        declared_valid = audit.get("num_valid_scenes")
        if declared_valid is not None and declared_valid != len(shard_values):
            raise ValueError(
                f"{path}: num_valid_scenes={declared_valid} but contains "
                f"{len(shard_values)} finite scenes"
            )
        declared_total = audit.get("num_total_scenes")
        if declared_total is not None and declared_total != len(scene_items):
            raise ValueError(
                f"{path}: num_total_scenes={declared_total} but per_scene_opw "
                f"contains {len(scene_items)} entries"
            )

    if not opw_by_scene:
        raise ValueError("OPW audit JSONs contain no finite per-scene values")

    merged = dict(audits[0])
    merged["per_scene_opw"] = [
        {"scene": scene, "opw": opw}
        for scene, opw in sorted(opw_by_scene.items())
    ]
    merged["mean_opw"] = sum(opw_by_scene.values()) / len(opw_by_scene)
    merged["num_valid_scenes"] = len(opw_by_scene)
    merged["num_total_scenes"] = sum(
        int(audit.get("num_total_scenes", 0)) for audit in audits
    )
    if merged["num_total_scenes"] != len(opw_by_scene):
        raise ValueError(
            "OPW audit scene counts are inconsistent: "
            f"num_total_scenes={merged['num_total_scenes']}, "
            f"finite unique scenes={len(opw_by_scene)}"
        )
    return merged, opw_by_scene


def merge_summary(
    summary_path: Path,
    opw_path: Path | list[Path],
    output_path: Path | None,
) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    opw_paths = [opw_path] if isinstance(opw_path, Path) else list(opw_path)
    audit, opw_by_scene = _merge_audits(opw_paths)

    mean_opw = audit.get("mean_opw")
    if (
        isinstance(mean_opw, bool)
        or not isinstance(mean_opw, (int, float))
        or not math.isfinite(mean_opw)
    ):
        raise ValueError(f"mean_opw must be a finite number, got {mean_opw!r}")
    expected_mean = sum(opw_by_scene.values()) / len(opw_by_scene)
    if not math.isclose(float(mean_opw), expected_mean, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"mean_opw {mean_opw!r} disagrees with per-scene mean {expected_mean!r}"
        )

    list_key, scene_items, summary_scenes = _summary_scene_items(summary)
    summary_scene_set = set(summary_scenes)
    opw_scene_set = set(opw_by_scene)
    missing = sorted(summary_scene_set - opw_scene_set)
    extra = sorted(opw_scene_set - summary_scene_set)
    if missing or extra:
        raise ValueError(
            "Summary and OPW audit scene coverage differ: "
            f"missing OPW={missing}, extra OPW={extra}"
        )
    declared_samples = summary.get("num_samples")
    if declared_samples is not None and declared_samples != len(summary_scenes):
        raise ValueError(
            f"summary num_samples={declared_samples} but {list_key} has "
            f"{len(summary_scenes)} entries"
        )

    avg_metrics = summary.setdefault("avg_metrics", {})
    if not isinstance(avg_metrics, dict):
        raise ValueError(f"{summary_path}: avg_metrics must be an object")
    previous_opw = avg_metrics.get("opw")
    avg_metrics["opw"] = float(mean_opw)
    per_scene_updated = _update_per_scene(scene_items, summary_scenes, opw_by_scene)

    summary.pop("opw_audit", None)
    summary["opw_evaluation"] = {
        "status": "computed_offline",
        "online": False,
        "source_json": (
            str(opw_paths[0])
            if len(opw_paths) == 1
            else [str(path) for path in opw_paths]
        ),
        "mean_opw": float(mean_opw),
        "num_valid_scenes": audit.get("num_valid_scenes"),
        "num_total_scenes": audit.get("num_total_scenes"),
        "gmflow_ckpt": audit.get("gmflow_ckpt"),
        "gmflow_repo": audit.get("gmflow_repo"),
        "beta": audit.get("beta"),
        "fb_consistency": audit.get("fb_consistency"),
        "protocol": audit.get("protocol"),
        "depth_key": audit.get("depth_key"),
        "eval_mask_key": audit.get("eval_mask_key"),
        "flow_batch_size": audit.get("flow_batch_size"),
        "flow_max_side": audit.get("flow_max_side"),
    }

    destination = output_path or summary_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() == summary_path.resolve():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = summary_path.with_name(f"{summary_path.name}.before_opw_{timestamp}")
        shutil.copy2(summary_path, backup)
        print(f"backup: {backup}")

    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)

    print(f"summary: {destination}")
    print(f"avg_metrics.opw: {previous_opw!r} -> {float(mean_opw):.6f}")
    print(f"per-scene entries updated: {per_scene_updated}/{len(opw_by_scene)}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--opw-json", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. By default update --summary after making a backup.",
    )
    args = parser.parse_args()

    if not args.summary.is_file():
        raise FileNotFoundError(args.summary)
    for path in args.opw_json:
        if not path.is_file():
            raise FileNotFoundError(path)
    merge_summary(args.summary, args.opw_json, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
