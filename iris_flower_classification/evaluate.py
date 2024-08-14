"""Evaluation phase of the model training."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(y_test: pd.Series, y_pred: pd.Series) -> dict:
    """Evaluate classification and returns metrics."""

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_avg_precision": precision_score(
            y_test,
            y_pred,
            average="macro",
        ),
        "macro_avg_recall": recall_score(y_test, y_pred, average="macro"),
    }
