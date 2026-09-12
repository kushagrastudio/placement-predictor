import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_importance(fitted_pipeline, X_test, y_test, top_n=10):
    """Uses permutation importance — model-agnostic, easiest to defend in viva
    (works the same way regardless of whether it's Logistic Regression or
    Random Forest underneath)."""
    result = permutation_importance(
        fitted_pipeline, X_test, y_test, n_repeats=10, random_state=42
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return importance_df.head(top_n)
