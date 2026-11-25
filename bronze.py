
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "bronze" / "raw_transactions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "bronze" / "bronze_transactions.csv"
REPORT_FILE = BASE_DIR / "reports" / "bronze_report.json"

REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_amount",
    "transaction_type",
    "status",
    "channel",
    "country",
    "transaction_timestamp"
]


def run_bronze():

    print("=" * 55)
    print("BRONZE LAYER - RAW DATA INGESTION")
    print("=" * 55)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run generate_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Schema validation: PASS")

    # Capture initial data quality metrics
    report = {
        "layer": "bronze",
        "source_file": INPUT_FILE.name,
        "total_records": int(len(df)),
        "duplicate_records": int(
            df.duplicated(subset=["transaction_id"]).sum()
        ),
        "missing_customer_ids": int(
            df["customer_id"].isna().sum()
        ),
        "negative_amounts": int(
            (df["transaction_amount"] < 0).sum()
        ),
        "invalid_statuses": int(
            (~df["status"].isin(
                ["SUCCESS", "FAILED", "PENDING"]
            )).sum()
        ),
        "missing_timestamps": int(
            df["transaction_timestamp"].isna().sum()
        ),
        "high_value_transactions": int(
            (df["transaction_amount"] >= 50000).sum()
        ),
        "ingestion_timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }

    # Add ingestion metadata
    df["source_file"] = INPUT_FILE.name
    df["ingestion_timestamp"] = report["ingestion_timestamp"]

    # Save Bronze data
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Save ingestion report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Display results
    print(f"Total records ingested : {len(df):,}")
    print(f"Duplicate records      : {report['duplicate_records']:,}")
    print(f"Missing customer IDs   : {report['missing_customer_ids']:,}")
    print(f"Negative amounts       : {report['negative_amounts']:,}")
    print(f"Invalid statuses       : {report['invalid_statuses']:,}")
    print(f"Missing timestamps     : {report['missing_timestamps']:,}")
    print(f"High-value transactions: {report['high_value_transactions']:,}")

    print("-" * 55)
    print(f"Bronze data saved: {OUTPUT_FILE}")
    print(f"Report saved: {REPORT_FILE}")
    print("BRONZE INGESTION COMPLETED")
    print("=" * 55)

    return report


if __name__ == "__main__":
    run_bronze()
