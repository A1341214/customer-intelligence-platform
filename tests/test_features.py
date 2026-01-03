import pandas as pd
from pathlib import Path

FEATURE_DATA = Path("data/processed/features.csv")


def test_feature_file_exists():
    assert FEATURE_DATA.exists(), "Feature file not found"


def test_no_null_features():
    df = pd.read_csv(FEATURE_DATA)
    assert df.isnull().sum().sum() == 0, "Null values found in features"


def test_churn_binary():
    df = pd.read_csv(FEATURE_DATA)
    assert set(df["churn"].unique()).issubset({0, 1}), "Churn is not binary"
