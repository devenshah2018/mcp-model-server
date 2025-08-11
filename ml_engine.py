import os
import json
import uuid
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

ALGO_MAP = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "mlp": MLPClassifier
}

def _model_path(model_id):
    return os.path.join(MODELS_DIR, f"{model_id}.joblib")

def _meta_path(model_id):
    return os.path.join(MODELS_DIR, f"{model_id}.meta.json")

def list_datasets():
    files = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
    return files

def load_dataset_csv(name):
    path = os.path.join(DATASETS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df

def train_model(dataset: str, algorithm: str = "logistic_regression", target_col: str = "target",
                test_size: float = 0.2, random_state: int = 42, hyperparams: Optional[Dict[str, Any]] = None):
    if algorithm not in ALGO_MAP:
        raise ValueError(f"unknown algorithm {algorithm}")
    df = load_dataset_csv(dataset)
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col} not in dataset columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    ModelClass = ALGO_MAP[algorithm]
    hyperparams = hyperparams or {}
    if algorithm == "logistic_regression":
        hyperparams.setdefault("max_iter", 300)
        hyperparams.setdefault("solver", "lbfgs")
        hyperparams.setdefault("multi_class", "auto")
    model = ModelClass(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _calc_metrics(y_test, y_pred)
    model_id = f"{algorithm}_{uuid.uuid4().hex[:8]}"
    joblib.dump(model, _model_path(model_id))
    meta = {
        "model_id": model_id,
        "algorithm": algorithm,
        "dataset": dataset,
        "target_col": target_col,
        "hyperparams": hyperparams,
        "metrics": metrics
    }
    with open(_meta_path(model_id), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta

def _calc_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }

def load_model(model_id: str):
    path = _model_path(model_id)
    meta_path = _meta_path(model_id)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = joblib.load(path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    return model, meta

def evaluate_model(model_id: str, dataset: Optional[str] = None, target_col: Optional[str] = None):
    model, meta = load_model(model_id)
    dataset = dataset or meta.get("dataset")
    target_col = target_col or meta.get("target_col", "target")
    if not dataset:
        raise ValueError("dataset not specified and model metadata doesn't contain dataset")
    df = load_dataset_csv(dataset)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    y_pred = model.predict(X)
    metrics = _calc_metrics(y, y_pred)
    meta["metrics"] = metrics
    with open(_meta_path(model_id), "w") as fh:
        json.dump(meta, fh, indent=2)
    return metrics

def predict(model_id: str, input_list: List[List[float]]):
    model, meta = load_model(model_id)
    arr = np.array(input_list)
    pred = model.predict(arr).tolist()
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(arr).tolist()
    return {"predictions": pred, "probabilities": proba}

def explain_prediction(model_id: str, input_row: List[float], dataset_for_baseline: Optional[str] = None, target_col: Optional[str] = None, n_repeats: int = 5):
    model, meta = load_model(model_id)
    dataset = dataset_for_baseline or meta.get("dataset")
    target_col = target_col or meta.get("target_col", "target")
    if not dataset:
        raise ValueError("dataset needed for baseline to compute permutation importance")
    df = load_dataset_csv(dataset)
    X = df.drop(columns=[target_col]).values
    feature_names = list(df.drop(columns=[target_col]).columns)
    if hasattr(model, "coef_"):
        coefs = np.mean(np.abs(model.coef_), axis=0).tolist()
        paired = list(zip(feature_names, coefs))
        return {"feature_names": feature_names, "importances": coefs, "method": "coef", "paired": paired}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.tolist()
        paired = list(zip(feature_names, fi))
        return {"feature_names": feature_names, "importances": fi, "method": "feature_importances", "paired": paired}
    try:
        r = permutation_importance(model, X, df[target_col].values, n_repeats=n_repeats, random_state=0)
        importances = r.importances_mean.tolist()
        paired = list(zip(feature_names, importances))
        return {"feature_names": feature_names, "importances": importances, "method": "permutation", "paired": paired}
    except Exception as e:
        return {"error": str(e)}

def list_models():
    metas = []
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".meta.json"):
            with open(os.path.join(MODELS_DIR, fname)) as fh:
                metas.append(json.load(fh))
    return metas

def export_model(model_id: str, export_format: str = "joblib"):
    if export_format != "joblib":
        raise ValueError("only joblib export supported")
    src = _model_path(model_id)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    dst = os.path.join(MODELS_DIR, f"{model_id}.export.{export_format}")
    joblib.dump(joblib.load(src), dst)
    return {"export_path": dst}
