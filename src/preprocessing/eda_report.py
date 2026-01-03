import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/customer_data.csv")
REPORT_PATH = Path("data/processed/eda_summary.csv")


def run_eda(df: pd.DataFrame):
    print("\n📊 BASIC INFO")
    print(df.info())

    print("\n📈 DESCRIPTIVE STATS")
    print(df.describe(include="all"))

    print("\n❓ MISSING VALUES")
    print(df.isnull().sum())

    print("\n🎯 CHURN DISTRIBUTION")
    print(df["churn"].value_counts(normalize=True))

    summary = pd.DataFrame({
        "missing_pct": df.isnull().mean(),
        "unique_values": df.nunique()
    })

    return summary


def main():
    df = pd.read_csv(DATA_PATH)
    summary = run_eda(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_PATH)

    print(f"\n✅ EDA summary saved to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
