import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from iris_flower_classification.data import dump_data, load_data


def prepare(X, y):
    y = pd.Categorical(y).codes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21
    )

    return X_train, X_test, y_train, y_test


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset-path", "-r")
    parser.add_argument("--prepared-data-dir", "-p")

    args = parser.parse_args()
    return args.raw_dataset_path, args.prepared_data_dir


def _check_is_dir(path):
    if not os.path.isdir(path):
        raise NotADirectoryError()


def main():
    raw_dataset_path, prepared_data_dir = _get_arguments()
    _check_is_dir(prepared_data_dir)

    X, y = load_data(raw_dataset_path)
    X_train, X_test, y_train, y_test = prepare(X, y)

    dump_data(X_train, y_train, path=f"{prepared_data_dir}/train.csv")
    dump_data(X_test, y_test, path=f"{prepared_data_dir}/test.csv")
