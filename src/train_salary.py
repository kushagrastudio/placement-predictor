import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from load_data import load_raw_data
from preprocessing import build_preprocessor, get_salary_features_target

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_and_compare():
    df = load_raw_data()
    X, y = get_salary_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest_reg": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"mae": mae, "r2": r2}
        fitted_pipelines[name] = pipeline

        print(f"{name}: MAE={mae:.4f}, R2={r2:.4f}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_pipeline, MODEL_DIR / "salary_regressor.pkl")
    print(f"Saved best model to {MODEL_DIR / 'salary_regressor.pkl'}")

    return results


if __name__ == "__main__":
    train_and_compare()
