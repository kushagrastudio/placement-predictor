
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Columns to drop entirely — no predictive value
DROP_COLS = ["student_id"]

# Nominal categorical columns -> one-hot encoding (no natural order)
# NOTE: confirm exact category values once EDA output is in (e.g. does
# college_tier need ordinal encoding instead? check its actual values).
CATEGORICAL_COLS = ["gender", "branch", "college_tier", "volunteer_experience"]

# Numeric columns -> imputed (median) + scaled
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


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning before splitting features/target."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates()
    return df


def build_preprocessor() -> ColumnTransformer:
    """Returns an UNFITTED preprocessor. Must be .fit() only on training
    data, never on the full dataset, to avoid leakage."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])
    return preprocessor


def get_placement_features_target(df: pd.DataFrame):
    """Returns X, y for the classification task."""
    df = clean_raw_data(df)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_CLASSIFICATION]
    return X, y


def get_salary_features_target(df: pd.DataFrame):
    """Returns X, y for the regression task — ONLY placed students,
    per our architectural decision (salary is undefined otherwise)."""
    df = clean_raw_data(df)
    placed_df = df[df[TARGET_CLASSIFICATION].astype(str).str.lower().isin(["placed", "1", "yes", "true"])]
    X = placed_df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = placed_df[TARGET_REGRESSION]
    return X, y