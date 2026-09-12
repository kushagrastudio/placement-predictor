import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/student_placement_dataset.csv")


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. "
            "Download it from Kaggle and place it there."
        )
    return pd.read_csv(RAW_DATA_PATH)


if _name_ == "_main_":
    df = load_raw_data()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")