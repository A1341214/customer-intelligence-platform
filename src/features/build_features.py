import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/raw/customer_data.csv")
PROCESSED_DATA = Path("data/processed/features.csv")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Monetary intensity
    df["avg_charge_per_month"] = df["total_charges"] / (df["tenure_months"] + 1)

    # Engagement score
    df["engagement_score"] = (
        df["usage_minutes"] / df["usage_minutes"].max()
        - df["last_login_days"] / df["last_login_days"].max()
    )

    # Support burden
    df["support_rate"] = df["support_tickets"] / (df["tenure_months"] + 1)

    # Contract encoding
    df["is_monthly_contract"] = (df["contract_type"] == "Month-to-Month").astype(int)

    # High value customer flag
    df["high_value_customer"] = (df["total_charges"] > 5000).astype(int)

    return df


def main():
    df = pd.read_csv(RAW_DATA)
    df_features = create_features(df)

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(PROCESSED_DATA, index=False)

    print(f"✅ Feature dataset saved to {PROCESSED_DATA.resolve()}")


if __name__ == "__main__":
    main()
