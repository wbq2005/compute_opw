from pathlib import Path

import pytest

from scripts.audit_opw_metric import (
    _collect_pairs,
    _resolve_output_paths,
    _write_tsv,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_collect_pairs_requires_exact_coverage(tmp_path):
    input_dir = tmp_path / "input"
    pred_dir = tmp_path / "pred"
    _touch(input_dir / "scene_000.pt")
    _touch(input_dir / "scene_001.pt")
    _touch(pred_dir / "scene_000_pred.pt")

    with pytest.raises(FileNotFoundError, match="missing predictions=.*scene_001"):
        _collect_pairs(input_dir, pred_dir)


def test_collect_pairs_returns_sorted_exact_matches(tmp_path):
    input_dir = tmp_path / "input"
    pred_dir = tmp_path / "pred"
    for scene in ("scene_001", "scene_000"):
        _touch(input_dir / f"{scene}.pt")
        _touch(pred_dir / f"{scene}_pred.pt")

    pairs = _collect_pairs(input_dir, pred_dir)

    assert [scene for scene, _, _ in pairs] == ["scene_000", "scene_001"]


def test_write_tsv_uses_atomic_replacement(tmp_path):
    output = tmp_path / "opw.tsv"

    _write_tsv(output, [{"scene": "scene_000", "opw": 1.25}])

    assert output.read_text(encoding="utf-8") == "scene\topw\nscene_000\t1.250000\n"
    assert not (tmp_path / ".opw.tsv.tmp").exists()


def test_full_audit_defaults_to_artifacts_and_adjacent_summary(tmp_path):
    pred_dir = tmp_path / "pred"
    _touch(pred_dir / "summary.json")

    partial, out_json, out_tsv, summary = _resolve_output_paths(
        pred_dir, 0, None, None, None, None, False
    )

    assert partial is False
    assert out_json == pred_dir / "opw.json"
    assert out_tsv == pred_dir / "opw.tsv"
    assert summary == pred_dir / "summary.json"


def test_probe_uses_separate_artifacts_and_never_auto_merges(tmp_path):
    pred_dir = tmp_path / "pred"
    _touch(pred_dir / "summary.json")

    partial, out_json, out_tsv, summary = _resolve_output_paths(
        pred_dir, 3, 1, None, None, None, False
    )

    assert partial is True
    assert out_json == pred_dir / "opw_probe_offset3_n1.json"
    assert out_tsv == pred_dir / "opw_probe_offset3_n1.tsv"
    assert summary is None
