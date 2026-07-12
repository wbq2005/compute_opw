from pathlib import Path

import pytest

from scripts.audit_opw_metric import _collect_pairs, _write_tsv


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
