from iris_flower_classification.data import load_data


def main():
    X_train, y_train = load_data(dataset_path)
    model = train(X_train, y_train)
    save_model(model)
