import argparse

import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from iris_flower_classification.data import load_data


def reduce_dimensionality(X):
    principal = PCA(n_components=2)
    principal.fit(X)

    return principal.transform(X)


def plot(X, y, figure_path):
    x = reduce_dimensionality(X)
    iris_data = pd.DataFrame(x, columns=["PC1", "PC2"])
    iris_data["class"] = y
    iris_data["class"] = iris_data["class"].apply(label_num_to_str)
    scatter_plot = sns.scatterplot(iris_data, x="PC1", y="PC2", hue="class")
    scatter_plot.figure.savefig(figure_path)


def label_num_to_str(label_num: int) -> str:
    label_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    return label_dict[label_num]


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")
    parser.add_argument("--figure-path")

    return parser.parse_args()


def main():
    args = _parse_arguments()
    X, y = load_data(args.dataset_path)
    plot(X, y, args.figure_path)
