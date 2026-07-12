import json

import pytest

from scripts.merge_opw_summary import merge_summary


def test_merge_opw_summary_updates_average_and_matching_scenes(tmp_path):
    summary_path = tmp_path / "summary.json"
    opw_path = tmp_path / "opw.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {"scene": "scene_000", "metrics": {"absrel": 0.1, "opw": None}},
                    {"scene": "scene_001", "metrics": {"absrel": 0.2, "opw": None}},
                ],
                "avg_metrics": {"absrel": 0.15, "opw": float("nan")},
            }
        ),
        encoding="utf-8",
    )
    opw_path.write_text(
        json.dumps(
            {
                "mean_opw": 149.4,
                "per_scene_opw": [
                    {"scene": "scene_000", "opw": 100.0},
                    {"scene": "scene_001", "opw": 198.8},
                ],
                "num_valid_scenes": 2,
                "num_total_scenes": 2,
                "fb_consistency": True,
                "opw_mode": "capa_strict",
                "flow_batch_size": 2,
            }
        ),
        encoding="utf-8",
    )

    merge_summary(summary_path, opw_path, None)

    merged = json.loads(summary_path.read_text(encoding="utf-8"))
    assert merged["avg_metrics"]["opw"] == 149.4
    assert merged["results"][0]["metrics"]["opw"] == 100.0
    assert merged["results"][1]["metrics"]["opw"] == 198.8
    assert merged["opw_evaluation"]["status"] == "computed_offline"
    assert merged["opw_evaluation"]["fb_consistency"] is True
    assert "opw_audit" not in merged
    assert "NaN" not in summary_path.read_text(encoding="utf-8")
    assert len(list(tmp_path.glob("summary.json.before_opw_*"))) == 1


def test_merge_opw_summary_combines_disjoint_shards(tmp_path):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "merged.json"
    shard_a = tmp_path / "shard_a.json"
    shard_b = tmp_path / "shard_b.json"
    summary_path.write_text(
        json.dumps(
            {
                "per_sample": [
                    {"sample": "scene_000", "opw": None},
                    {"sample": "scene_001", "opw": None},
                ],
                "avg_metrics": {"opw": None},
            }
        ),
        encoding="utf-8",
    )
    common = {"fb_consistency": True, "opw_mode": "capa_strict"}
    shard_a.write_text(
        json.dumps(
            {
                **common,
                "per_scene_opw": [{"scene": "scene_000", "opw": 100.0}],
                "num_total_scenes": 1,
            }
        ),
        encoding="utf-8",
    )
    shard_b.write_text(
        json.dumps(
            {
                **common,
                "per_scene_opw": [{"scene": "scene_001", "opw": 200.0}],
                "num_total_scenes": 1,
            }
        ),
        encoding="utf-8",
    )

    merge_summary(summary_path, [shard_a, shard_b], output_path)

    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert merged["avg_metrics"]["opw"] == 150.0
    assert merged["per_sample"][0]["opw"] == 100.0
    assert merged["per_sample"][1]["opw"] == 200.0
    assert merged["opw_evaluation"]["source_json"] == [str(shard_a), str(shard_b)]
    assert merged["opw_evaluation"]["num_valid_scenes"] == 2


def test_merge_opw_summary_rejects_duplicate_scenes_across_shards(tmp_path):
    summary_path = tmp_path / "summary.json"
    shard_a = tmp_path / "shard_a.json"
    shard_b = tmp_path / "shard_b.json"
    summary_path.write_text(json.dumps({"avg_metrics": {}}), encoding="utf-8")
    payload = {
        "per_scene_opw": [{"scene": "scene_000", "opw": 1.0}],
        "num_total_scenes": 1,
    }
    shard_a.write_text(json.dumps(payload), encoding="utf-8")
    shard_b.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate scene"):
        merge_summary(summary_path, [shard_a, shard_b], None)


@pytest.mark.parametrize(
    ("summary_scenes", "opw_scenes"),
    [
        (["scene_000", "scene_001"], ["scene_000"]),
        (["scene_000"], ["scene_000", "scene_001"]),
    ],
)
def test_merge_opw_summary_requires_exact_scene_coverage(
    tmp_path, summary_scenes, opw_scenes
):
    summary_path = tmp_path / "summary.json"
    opw_path = tmp_path / "opw.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_samples": len(summary_scenes),
                "per_sample": [
                    {"sample": scene, "absrel": 0.1} for scene in summary_scenes
                ],
                "avg_metrics": {"absrel": 0.1},
            }
        ),
        encoding="utf-8",
    )
    values = [float(index + 1) for index in range(len(opw_scenes))]
    opw_path.write_text(
        json.dumps(
            {
                "mean_opw": sum(values) / len(values),
                "per_scene_opw": [
                    {"scene": scene, "opw": value}
                    for scene, value in zip(opw_scenes, values)
                ],
                "num_valid_scenes": len(opw_scenes),
                "num_total_scenes": len(opw_scenes),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scene coverage differ"):
        merge_summary(summary_path, opw_path, None)


def test_merge_opw_summary_rejects_inconsistent_mean(tmp_path):
    summary_path = tmp_path / "summary.json"
    opw_path = tmp_path / "opw.json"
    summary_path.write_text(
        json.dumps(
            {
                "per_sample": [{"sample": "scene_000"}],
                "avg_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    opw_path.write_text(
        json.dumps(
            {
                "mean_opw": 2.0,
                "per_scene_opw": [{"scene": "scene_000", "opw": 1.0}],
                "num_valid_scenes": 1,
                "num_total_scenes": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagrees with per-scene mean"):
        merge_summary(summary_path, opw_path, None)


def test_merge_opw_summary_rejects_nonstandard_audit_json(tmp_path):
    summary_path = tmp_path / "summary.json"
    opw_path = tmp_path / "opw.json"
    summary_path.write_text(
        json.dumps({"per_sample": [{"sample": "scene_000"}], "avg_metrics": {}}),
        encoding="utf-8",
    )
    opw_path.write_text(
        '{"mean_opw": NaN, "per_scene_opw": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        merge_summary(summary_path, opw_path, None)


def test_merge_opw_summary_rejects_shard_protocol_mismatch(tmp_path):
    summary_path = tmp_path / "summary.json"
    shard_a = tmp_path / "shard_a.json"
    shard_b = tmp_path / "shard_b.json"
    summary_path.write_text(
        json.dumps({"per_sample": [], "avg_metrics": {}}),
        encoding="utf-8",
    )
    shard_a.write_text(
        json.dumps(
            {
                "fb_consistency": True,
                "per_scene_opw": [{"scene": "scene_000", "opw": 1.0}],
                "num_total_scenes": 1,
            }
        ),
        encoding="utf-8",
    )
    shard_b.write_text(
        json.dumps(
            {
                "per_scene_opw": [{"scene": "scene_001", "opw": 1.0}],
                "num_total_scenes": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagree on 'fb_consistency'"):
        merge_summary(summary_path, [shard_a, shard_b], None)
