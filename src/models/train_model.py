import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/processed/features.csv")
MODEL_DIR = Path("models")

TARGET = "churn"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_pipeline(model):
    categorical_features = ["contract_type", "payment_method"]
    numerical_features = [
        col for col in model.feature_names_in_
        if col not in categorical_features
    ] if hasattr(model, "feature_names_in_") else None

    preprocessor = ColumnTransformer(
        transformers=[
           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["contract_type", "payment_method"])
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    pipeline = build_pipeline(model)

    pipeline.fit(X_train, y_train)

    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\n📊 {model_name} ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / f"{model_name}.joblib")

    print(f"✅ Model saved: models/{model_name}.joblib")


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🚀 Training Logistic Regression (Baseline)")
    lr_model = LogisticRegression(max_iter=1000)
    train_and_evaluate(lr_model, X_train, X_test, y_train, y_test, "logistic_regression")

    print("\n🚀 Training Random Forest (Advanced)")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    train_and_evaluate(rf_model, X_train, X_test, y_train, y_test, "random_forest")


if __name__ == "__main__":
    main()
