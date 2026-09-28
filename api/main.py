from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

def load_model(path):
    return joblib.load(path)

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

@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    try:
        placement_model = load_model(MODEL_DIR / "placement_classifier.pkl")
        salary_model = load_model(MODEL_DIR / "salary_regressor.pkl")
        
        student_data = student.model_dump()
        X = pd.DataFrame([student_data])

        if int(student_data.get("backlogs", 0)) > 0:
            return {
                "placement_prediction": "Not Placed",
                "placement_probability": 1.0,
                "salary_prediction": None,
                "note": "Marked ineligible due to active backlogs.",
            }

        placement_pred = placement_model.predict(X)[0]
        classes = list(placement_model.classes_)
        placement_proba = placement_model.predict_proba(X)[0][classes.index("Placed")]

        result = {
            "placement_prediction": placement_pred,
            "placement_probability": round(float(placement_proba), 4),
            "salary_prediction": None,
            "note": None,
        }
        if placement_pred == "Placed":
            result["salary_prediction"] = round(float(salary_model.predict(X)[0]), 2)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
