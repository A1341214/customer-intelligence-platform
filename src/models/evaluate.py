import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/random_forest.joblib")


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    model = joblib.load(MODEL_PATH)

    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    print("📊 Evaluation Metrics")
    print("ROC-AUC:", roc_auc_score(y, y_pred_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    main()
