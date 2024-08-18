import argparse

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from iris_flower_classification.data import load_data


def reduce_dimensionality(X):
    principal = PCA(n_components=2)
    principal.fit(X)

    return principal.transform(X)


def plot(X, y, figure_path):
    x = reduce_dimensionality(X)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="plasma")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(figure_path)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")
    parser.add_argument("--figure-path")

    return parser.parse_args()


def main():
    args = _parse_arguments()
    X, y = load_data(args.dataset_path)
    plot(X, y, args.figure_path)
