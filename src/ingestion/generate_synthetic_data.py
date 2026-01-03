import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_CUSTOMERS = 5000
OUTPUT_PATH = Path("data/raw/customer_data.csv")

def generate_customer_data(n_customers: int) -> pd.DataFrame:
    customer_id = np.arange(1, n_customers + 1)

    age = np.random.randint(18, 70, n_customers)
    tenure_months = np.random.randint(1, 60, n_customers)
    monthly_charges = np.round(np.random.uniform(10, 150, n_customers), 2)
    total_charges = np.round(monthly_charges * tenure_months, 2)

    usage_minutes = np.random.normal(500, 120, n_customers).clip(50, 2000)
    support_tickets = np.random.poisson(1.2, n_customers)
    last_login_days = np.random.randint(0, 60, n_customers)

    contract_type = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_customers,
        p=[0.6, 0.25, 0.15]
    )

    payment_method = np.random.choice(
        ["Credit Card", "Debit Card", "UPI", "Net Banking"],
        size=n_customers
    )

    # Churn logic (non-random, business-driven)
    churn_probability = (
        0.4 * (contract_type == "Month-to-Month").astype(int) +
        0.3 * (support_tickets > 3).astype(int) +
        0.2 * (last_login_days > 30).astype(int) +
        0.2 * (monthly_charges > 100).astype(int)
    )

    churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_customers), 0, 1)
    churn = np.random.binomial(1, churn_probability)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "usage_minutes": usage_minutes.astype(int),
        "support_tickets": support_tickets,
        "last_login_days": last_login_days,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "churn": churn
    })

    return df


def save_data(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic data saved to: {output_path.resolve()}")


if __name__ == "__main__":
    data = generate_customer_data(N_CUSTOMERS)
    save_data(data, OUTPUT_PATH)
