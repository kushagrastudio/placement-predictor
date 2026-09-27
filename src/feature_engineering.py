"""
feature_engineering.py
Based on real EDA: no redundant features found among the score columns;
no feature correlates strongly with placement_status. See docs/eda_findings.md.
"""
import pandas as pd


def add_backlog_eligible_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit POLICY column (backlogs == 0 -> eligible), not a learned pattern."""
    df = df.copy()
    df["backlog_eligible"] = df["backlogs"] == 0
    return df
