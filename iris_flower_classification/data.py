from pathlib import Path

import pandas as pd


def load_data(dataset_path: Path) -> tuple[pd.Series, pd.Series]:
    iris_data = pd.read_csv(dataset_path)
    X = iris_data.drop("class", axis=1).values
    y = iris_data["class"].values

    return X, y
