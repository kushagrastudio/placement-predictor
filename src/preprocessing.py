"""
preprocessing.py
Final version, based on real EDA findings (100,000 rows):
- college_tier is ORDINAL -> ordinal encoding
- gender, branch, volunteer_experience are NOMINAL -> one-hot encoding
- salary_package_lpa EXCLUDED from classifier features (leakage)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DROP_COLS = ["student_id"]
NOMINAL_COLS = ["gender", "branch", "volunteer_experience"]
ORDINAL_COLS = ["college_tier"]
ORDINAL_CATEGORIES = [["Tier 3", "Tier 2", "Tier 1"]]

NUMERIC_COLS = [
    "age", "cgpa", "internships_count", "projects_count",
    "certifications_count", "coding_skill_score", "aptitude_score",
    "communication_skill_score", "logical_reasoning_score",
    "hackathons_participated", "github_repos", "linkedin_connections",
    "mock_interview_score", "attendance_percentage", "backlogs",
    "extracurricular_score", "leadership_score", "sleep_hours",
    "study_hours_per_day",
]

TARGET_CLASSIFICATION = "placement_status"
TARGET_REGRESSION = "salary_package_lpa"
ALL_FEATURE_COLS = NUMERIC_COLS + NOMINAL_COLS + ORDINAL_COLS


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df.drop_duplicates()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    nominal_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    ordinal_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1))])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("nom", nominal_pipeline, NOMINAL_COLS),
        ("ord", ordinal_pipeline, ORDINAL_COLS),
    ])


def get_placement_features_target(df: pd.DataFrame):
    df = clean_raw_data(df)
    return df[ALL_FEATURE_COLS], df[TARGET_CLASSIFICATION]


def get_salary_features_target(df: pd.DataFrame):
    df = clean_raw_data(df)
    placed_df = df[df[TARGET_CLASSIFICATION] == "Placed"]
    return placed_df[ALL_FEATURE_COLS], placed_df[TARGET_REGRESSION]
