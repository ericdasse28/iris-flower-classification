from math import isclose

import pandas as pd

from iris_flower_classification.evaluate import evaluate


def test_evaluate_function_metrics():
    """Given original and predicted labels,
    When `evaluate` is called,
    Returns accuracy, macro average precision
    and macro average recall."""

    y_test = pd.Series([0, 1, 1, 2, 0, 0])
    y_pred = pd.Series([1, 1, 0, 2, 0, 0])

    actual_metrics = evaluate(y_test, y_pred)

    expected_metrics = {
        "accuracy": 0.66,
        "macro_avg_precision": 0.72,
        "macro_avg_recall": 0.72,
    }
    assert all(
        isclose(actual_metrics[metric], expected_metrics[metric], rel_tol=0.05)
        for metric in expected_metrics.keys()
    )
