import pandas as pd

from iris_flower_classification.data import dump_data


def test_dump_data(tmp_path):
    """Given features, labels and a path,
    When `dump_data` is called,
    Then features and labels are saved to path as
    a CSV file."""

    X = pd.DataFrame(
        {
            "sepal_length": [2.5, 2.8, 3.9, 0.5],
            "sepal_width": [1.0, 2.0, 3.8, 3.9],
            "petal_length": [3.7, 2.9, 2.0, 2.9],
            "petal_width": [1.0, 2.4, 3.9, 3.9],
        }
    )
    y = pd.DataFrame({"class": [0, 1, 1, 2]})
    path = tmp_path / "data.csv"

    dump_data(X.values, y.values, path)

    actual_dumped_data = pd.read_csv(path)
    expected_dumped_data = pd.concat([X, y], axis=1)
    pd.testing.assert_frame_equal(actual_dumped_data, expected_dumped_data)
