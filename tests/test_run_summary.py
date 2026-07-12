import pytest

from run import _build_opw_evaluation_metadata


def _metadata(online_opw, avg_metrics, per_sample):
    return _build_opw_evaluation_metadata(
        online_opw=online_opw,
        avg_metrics=avg_metrics,
        per_sample=per_sample,
        gmflow_ckpt="/models/gmflow.pth",
        beta=50.0,
        fb_consistency=True,
        flow_batch_size=2,
        flow_max_side=None,
    )


def test_opw_metadata_marks_unrequested_metric_explicitly():
    metadata = _metadata(False, {"absrel": 0.1}, [{"absrel": 0.1}])

    assert metadata["status"] == "not_requested"
    assert metadata["online"] is False
    assert metadata["mean_opw"] is None
    assert metadata["num_valid_scenes"] is None
    assert metadata["gmflow_ckpt"] is None
    assert metadata["eval_mask_key"] is None


def test_opw_metadata_records_online_protocol_and_coverage():
    metadata = _metadata(
        True,
        {"absrel": 0.1, "opw": 2.25},
        [{"opw": 2.0}, {"opw": 2.5}, {"absrel": 0.2}],
    )

    assert metadata["status"] == "computed_online"
    assert metadata["mean_opw"] == 2.25
    assert metadata["num_valid_scenes"] == 2
    assert metadata["num_total_scenes"] == 2
    assert metadata["opw_mode"] == "capa_strict"
    assert metadata["depth_key"] == "depth_pred_nhw"
    assert metadata["eval_mask_key"] == "depth_gt_nvhw"
    assert metadata["fb_consistency"] is True


def test_opw_metadata_marks_single_frame_data_not_applicable():
    metadata = _metadata(True, {"absrel": 0.1}, [{"absrel": 0.1}])

    assert metadata["status"] == "not_applicable"
    assert metadata["num_valid_scenes"] == 0
    assert metadata["num_total_scenes"] == 0


def test_opw_metadata_rejects_inconsistent_average_and_per_sample_values():
    with pytest.raises(RuntimeError, match="summary is inconsistent"):
        _metadata(True, {"opw": 2.0}, [{"absrel": 0.1}])
