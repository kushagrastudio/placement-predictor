import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from load_data import load_raw_data
from preprocessing import build_preprocessor, get_placement_features_target

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_and_compare():
    df = load_raw_data()
    X, y = get_placement_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"accuracy": acc, "f1": f1}
        fitted_pipelines[name] = pipeline

        print(f"{name}: accuracy={acc:.4f}, f1={f1:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_pipeline, MODEL_DIR / "placement_classifier.pkl")
    print(f"Saved best model to {MODEL_DIR / 'placement_classifier.pkl'}")

    return results


if __name__ == "__main__":
    train_and_compare()