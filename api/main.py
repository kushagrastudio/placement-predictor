"""
api/main.py
Vercel serverless-compatible FastAPI entrypoint.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path
import pandas as pd

app = FastAPI(title="Placement & Salary Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_placement_model = None
_salary_model = None

def _load_models():
    global _placement_model, _salary_model
    if _placement_model is None:
        _placement_model = joblib.load(MODEL_DIR / "placement_classifier.pkl")
    if _salary_model is None:
        _salary_model = joblib.load(MODEL_DIR / "salary_regressor.pkl")

# ── Schemas ────────────────────────────────────────────────────
class StudentInput(BaseModel):
    age: int
    gender: str
    cgpa: float
    branch: str
    college_tier: str
    internships_count: int
    projects_count: int
    certifications_count: int
    coding_skill_score: float
    aptitude_score: float
    communication_skill_score: float
    logical_reasoning_score: float
    hackathons_participated: int
    github_repos: int
    linkedin_connections: int
    mock_interview_score: float
    attendance_percentage: float
    backlogs: int
    extracurricular_score: float
    leadership_score: float
    volunteer_experience: str
    sleep_hours: float
    study_hours_per_day: float

class PredictionOutput(BaseModel):
    placement_prediction: str
    placement_probability: float
    salary_prediction: Optional[float]
    note: Optional[str]

# ── Routes ─────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    try:
        _load_models()
        student_data = student.model_dump()
        X = pd.DataFrame([student_data])

        if int(student_data.get("backlogs", 0)) > 0:
            return {
                "placement_prediction": "Not Placed",
                "placement_probability": 1.0,
                "salary_prediction": None,
                "note": "Marked ineligible due to active backlogs.",
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
