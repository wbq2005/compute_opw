import math

from capa.utils.metric import average_metrics


def test_average_metrics_omits_keys_without_finite_values():
    averaged = average_metrics(
        [
            {"sample": "a", "absrel": 0.1, "opw": float("nan")},
            {"sample": "b", "absrel": 0.3, "opw": float("inf")},
        ],
        ignore_keys=["sample"],
    )

    assert averaged == {"absrel": 0.2}


def test_average_metrics_uses_only_finite_observations():
    averaged = average_metrics(
        [{"opw": 1.0}, {"opw": None}, {"opw": -math.inf}, {"opw": 3.0}]
    )

    assert averaged == {"opw": 2.0}
