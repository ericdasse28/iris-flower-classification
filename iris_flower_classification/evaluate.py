"""Evaluation phase of the model training."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from dvclive import Live
from iris_flower_classification.data import load_data


def load_model(model_path: Path):
    """Load model."""

    return joblib.load(model_path)


def compute_evaluation_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict:
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


def evaluate(model, X, y):
    y_pred = model.predict(X)
    metrics = compute_evaluation_metrics(y, y_pred)

    return metrics


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--stage")
    args = parser.parse_args()

    return args


def main():
    args = _get_arguments()
    model = load_model(args.model_path)
    X, y = load_data(args.dataset_path)

    metrics = evaluate(model, X, y)

    with Live(resume=True) as live:
        for metric in metrics:
            live.log_metric(f"{args.stage}/{metric}", metrics[metric])
