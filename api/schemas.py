"""
schemas.py
"""
from pydantic import BaseModel
from typing import Optional


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
