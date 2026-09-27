"""
main.py
Run: uvicorn api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import StudentInput, PredictionOutput
from src.predict import predict_student

app = FastAPI(title="Placement & Salary Predictor API")

# Allow the frontend (opened as a local file or on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    try:
        return predict_student(student.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
