from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_classifier(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def evaluate_regressor(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def run_cross_validation(pipeline, X, y, cv=5, scoring="f1_weighted"):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    return {"scores": scores.tolist(), "mean": scores.mean(), "std": scores.std()}
