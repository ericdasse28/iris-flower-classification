"""Evaluation phase of the model training."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from dvclive import Live
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_model(model_path: Path):
    """Load model."""

    return joblib.load(model_path)


def load_data(dataset_path: Path) -> tuple[pd.Series, pd.Series]:
    iris_data = pd.read_csv(dataset_path)
    X = iris_data.drop("class", axis=1).values
    y = iris_data["class"].values

    return X, y


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


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    args = parser.parse_args()

    return args.model_path, args.dataset_path


def main():
    model_path, dataset_path = _get_arguments()
    model = load_model(model_path)
    X_test, y_test = load_data(dataset_path)

    y_pred = model.fit(X_test)

    metrics = evaluate(y_test, y_pred)

    with Live(resume=True) as live:
        for metric in metrics:
            live.log_metric(f"test/{metric}", metric)
