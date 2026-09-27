"""
data_inspection.py
Run: python src/data_inspection.py
"""
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "student_placement_dataset.csv"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}.")
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> None:
    print("=" * 60); print("SHAPE"); print("=" * 60)
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print("\n" + "=" * 60); print("COLUMN TYPES"); print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60); print("MISSING VALUES"); print("=" * 60)
    missing_count = df.isnull().sum()
    print(missing_count[missing_count > 0] if missing_count.sum() > 0 else "No missing values found.")

    print("\n" + "=" * 60); print("DUPLICATE ROWS"); print("=" * 60)
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n" + "=" * 60); print("SUMMARY STATISTICS"); print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60); print("TARGET DISTRIBUTION: placement_status"); print("=" * 60)
    print(df["placement_status"].value_counts())

    print("\n" + "=" * 60); print("FIRST 5 ROWS"); print("=" * 60)
    print(df.head())


if __name__ == "__main__":
    inspect(load_data(RAW_DATA_PATH))
