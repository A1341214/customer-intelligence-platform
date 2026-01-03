import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

MODEL_PATH = Path("models/random_forest.joblib")
DATA_PATH = Path("data/processed/features.csv")
OUTPUT_DIR = Path("reports/explainability")

TARGET = "churn"


def main():
    # Load model and data
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])

    # Split pipeline
    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    # Transform features (dense, numeric)
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed_df)

    # --------- NORMALIZE SHAP OUTPUT (CRITICAL FIX) ---------
    if isinstance(shap_values, list):
        # Binary classification → take class 1 (churn)
        shap_values_to_plot = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        # New SHAP versions return a single array
        shap_values_to_plot = shap_values
        expected_value = explainer.expected_value

    # Sanity check (optional but useful)
    assert shap_values_to_plot.shape[1] == X_transformed_df.shape[1], \
        "SHAP values and feature matrix still misaligned"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------- Global explanation ---------
    shap.summary_plot(
        shap_values_to_plot,
        X_transformed_df,
        show=False
    )
    plt.savefig(
        OUTPUT_DIR / "global_feature_importance.png",
        bbox_inches="tight"
    )
    plt.close()

    # --------- Local explanation ---------
       

    idx = 10

    # SHAP values for this row
    local_values = shap_values_to_plot[idx]

    # Ensure 1D (handle multi-output edge case)
    if local_values.ndim > 1:
        local_values = local_values[:, 1]

    # ✅ Base value MUST be scalar
    if isinstance(expected_value, (list, tuple)) or hasattr(expected_value, "__len__"):
        base_value = expected_value[1]  # churn class
    else:
        base_value = expected_value

    shap.plots.waterfall(
        shap.Explanation(
            values=local_values,
            base_values=base_value,
            data=X_transformed_df.iloc[idx],
            feature_names=X_transformed_df.columns
        ),
        show=False
    )


    plt.savefig(
        OUTPUT_DIR / f"local_explanation_customer_{idx}.png",
        bbox_inches="tight"
    )
    plt.close()

    print("✅ SHAP explanations generated successfully")


if __name__ == "__main__":
    main()
