"""
predict.py
Applies the backlog eligibility POLICY before the model (see feature_engineering.py).
"""
import joblib
from pathlib import Path
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_placement_model = None
_salary_model = None


def _load_models():
    global _placement_model, _salary_model
    if _placement_model is None:
        _placement_model = joblib.load(MODEL_DIR / "placement_classifier.pkl")
    if _salary_model is None:
        _salary_model = joblib.load(MODEL_DIR / "salary_regressor.pkl")


def predict_student(student_data: dict) -> dict:
    _load_models()
    X = pd.DataFrame([student_data])

    if int(student_data.get("backlogs", 0)) > 0:
        return {
            "placement_prediction": "Not Placed",
            "placement_probability": 1.0,
            "salary_prediction": None,
            "note": "Marked ineligible due to active backlogs (project policy rule, not a model-derived pattern).",
        }

    placement_pred = _placement_model.predict(X)[0]
    classes = list(_placement_model.classes_)
    placement_proba = _placement_model.predict_proba(X)[0][classes.index("Placed")]

    result = {
        "placement_prediction": placement_pred,
        "placement_probability": round(float(placement_proba), 4),
        "salary_prediction": None,
        "note": None,
    }
    if placement_pred == "Placed":
        result["salary_prediction"] = round(float(_salary_model.predict(X)[0]), 2)
    return result
