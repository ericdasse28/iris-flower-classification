import os
from pathlib import Path

import pandas as pd


def load_data(dataset_path: Path) -> tuple[pd.Series, pd.Series]:
    iris_data = pd.read_csv(dataset_path)
    X = iris_data.drop("class", axis=1).values
    y = iris_data["class"].values

    return X, y


def dump_data(X: pd.Series, y: pd.Series, path: os.PathLike):
    iris_data = pd.DataFrame(
        X,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )
    iris_data["class"] = y

    iris_data.to_csv(path)
