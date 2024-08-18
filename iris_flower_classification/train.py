import argparse
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression

from iris_flower_classification.data import load_data


def train(X: pd.Series, y: pd.Series):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def save_model(model, model_path: Path):
    joblib.dump(model, model_path)


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")
    parser.add_argument("--model-path")
    args = parser.parse_args()

    return args.dataset_path, args.model_path


def main():
    logger.info("Model training...")
    dataset_path, model_path = _get_arguments()

    X_train, y_train = load_data(dataset_path)
    model = train(X_train, y_train)

    logger.info(f"Saving model to {model_path}...")
    save_model(model, model_path)
