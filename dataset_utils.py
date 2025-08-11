import os
import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

def save_sklearn_dataset(des, name):
    X = des.data
    y = des.target
    columns = list(des.feature_names) if hasattr(des, "feature_names") else [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False)
    return path

def create_sample_datasets():
    ensure_data_dir()
    iris = load_iris()
    wine = load_wine()
    iris_path = save_sklearn_dataset(iris, "iris.csv")
    wine_path = save_sklearn_dataset(wine, "wine.csv")
    print("Created sample datasets:")
    print(" -", iris_path)
    print(" -", wine_path)

if __name__ == "__main__":
    create_sample_datasets()
