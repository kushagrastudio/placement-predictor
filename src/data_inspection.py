import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/student_placement_dataset.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. "
            "Download the CSV from Kaggle and place it in data/raw/."
        )
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("COLUMN TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES (count and % per column)")
    print("=" * 60)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    missing_report = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    print(missing_report[missing_report["missing_count"] > 0])
    if missing_count.sum() == 0:
        print("No missing values found.")

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (numeric columns)")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("SUMMARY (categorical / object columns)")
    print("=" * 60)
    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe())
    else:
        print("No object-type columns detected.")

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION: placement_status")
    print("=" * 60)
    if "placement_status" in df.columns:
        print(df["placement_status"].value_counts())
        print(df["placement_status"].value_counts(normalize=True).round(3))
    else:
        print("Column 'placement_status' not found — check actual column name.")

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION: salary_package_lpa (placed students only)")
    print("=" * 60)
    if "salary_package_lpa" in df.columns:
        print(df["salary_package_lpa"].describe())
    else:
        print("Column 'salary_package_lpa' not found — check actual column name.")

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS (preview)")
    print("=" * 60)
    print(df.head())


if _name_ == "_main_":
    data = load_data(RAW_DATA_PATH)
    inspect(data)