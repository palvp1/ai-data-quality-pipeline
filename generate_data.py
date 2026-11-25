
from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
TOTAL_RECORDS = 50_000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "bronze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "raw_transactions.csv"

# Generate synthetic banking transactions
df = pd.DataFrame({
    "transaction_id": [
        f"TXN{i:08d}" for i in range(1, TOTAL_RECORDS + 1)
    ],
    "customer_id": [
        f"CUST{i:06d}"
        for i in rng.integers(1, 10001, TOTAL_RECORDS)
    ],
    "transaction_amount": np.round(
        rng.lognormal(5.5, 1.2, TOTAL_RECORDS), 2
    ),
    "transaction_type": rng.choice(
        ["DEPOSIT", "WITHDRAWAL", "TRANSFER", "PAYMENT"],
        TOTAL_RECORDS
    ),
    "status": rng.choice(
        ["SUCCESS", "FAILED", "PENDING"],
        TOTAL_RECORDS,
        p=[0.85, 0.10, 0.05]
    ),
    "channel": rng.choice(
        ["ATM", "MOBILE", "WEB", "BRANCH"],
        TOTAL_RECORDS
    ),
    "country": rng.choice(
        ["USA", "UK", "INDIA", "CANADA", "GERMANY"],
        TOTAL_RECORDS
    ),
    "transaction_timestamp": (
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(
            rng.integers(0, 86400 * 30, TOTAL_RECORDS),
            unit="s"
        )
    )
})

# Select separate records for each quality issue
indices = rng.permutation(TOTAL_RECORDS)

# 1. Missing customer IDs
df.loc[indices[:500], "customer_id"] = None

# 2. Negative transaction amounts
df.loc[indices[500:800], "transaction_amount"] *= -1

# 3. Invalid transaction statuses
df.loc[indices[800:1000], "status"] = "UNKNOWN"

# 4. Missing timestamps
df.loc[indices[1000:1200], "transaction_timestamp"] = pd.NaT

# 5. Unusually high transaction amounts
df.loc[indices[1200:1450], "transaction_amount"] = (
    np.round(rng.uniform(50000, 500000, 250), 2)
)

# 6. Duplicate valid transactions
duplicates = df.loc[indices[1450:1950]].copy()

df = pd.concat([df, duplicates], ignore_index=True)

# Shuffle records
df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# Save raw dataset
df.to_csv(OUTPUT_FILE, index=False)

# Generation report
print("=" * 55)
print("SYNTHETIC BANKING DATA GENERATION")
print("=" * 55)
print(f"Original transactions       : {TOTAL_RECORDS:,}")
print(f"Duplicate transactions      : {500:,}")
print(f"Total generated records     : {len(df):,}")
print("-" * 55)
print("INTENTIONALLY INTRODUCED ISSUES")
print("-" * 55)
print("Missing customer IDs        : 500")
print("Negative amounts            : 300")
print("Invalid statuses            : 200")
print("Missing timestamps          : 200")
print("Unusually high amounts      : 250")
print("Duplicate records           : 500")
print("-" * 55)
print(f"Dataset saved to: {OUTPUT_FILE}")
print("DATA GENERATION COMPLETED")
print("=" * 55)
