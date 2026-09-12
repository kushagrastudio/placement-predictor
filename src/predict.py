import joblib
from pathlib import Path
import pandas as pd

MODEL_DIR = Path("models")

placement_model = joblib.load(MODEL_DIR / "placement_classifier.pkl")
salary_model = joblib.load(MODEL_DIR / "salary_regressor.pkl")


def predict_student(student_data: dict) -> dict:
    """
    student_data: dict of raw feature values matching the dataset columns
    (excluding student_id, placement_status, salary_package_lpa).
    """
    X = pd.DataFrame([student_data])

    placement_pred = placement_model.predict(X)[0]
    placement_proba = placement_model.predict_proba(X)[0].max()

    is_placed = str(placement_pred).lower() in ["placed", "1", "yes", "true"]

    result = {
        "placement_prediction": placement_pred,
        "placement_probability": round(float(placement_proba), 4),
        "salary_prediction": None,
    }

    if is_placed:
        salary_pred = salary_model.predict(X)[0]
        result["salary_prediction"] = round(float(salary_pred), 2)

    return result