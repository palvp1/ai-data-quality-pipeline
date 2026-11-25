
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "bronze" / "bronze_transactions.csv"
OUTPUT_FILE = BASE_DIR / "data" / "silver" / "clean_transactions.csv"
QUARANTINE_FILE = BASE_DIR / "data" / "quarantine" / "rejected_transactions.csv"
REPORT_FILE = BASE_DIR / "reports" / "silver_report.json"

VALID_STATUSES = ["SUCCESS", "FAILED", "PENDING"]


def run_silver():

    print("=" * 55)
    print("SILVER LAYER - DATA QUALITY AND VALIDATION")
    print("=" * 55)

    # Check whether Bronze data exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Bronze data not found. Run bronze.py first."
        )

    # Load Bronze data
    df = pd.read_csv(INPUT_FILE)

    total_records = len(df)

    # Convert columns to appropriate data types
    df["transaction_amount"] = pd.to_numeric(
        df["transaction_amount"],
        errors="coerce"
    )

    df["transaction_timestamp"] = pd.to_datetime(
        df["transaction_timestamp"],
        errors="coerce"
    )

    # Define data quality validation rules
    rules = {
        "DUPLICATE_TRANSACTION": df.duplicated(
            subset=["transaction_id"],
            keep="first"
        ),

        "MISSING_TRANSACTION_ID": (
            df["transaction_id"].isna()
            | df["transaction_id"].fillna("").str.strip().eq("")
        ),

        "MISSING_CUSTOMER_ID": (
            df["customer_id"].isna()
            | df["customer_id"].fillna("").str.strip().eq("")
        ),

        "INVALID_AMOUNT": (
            df["transaction_amount"].isna()
            | ~np.isfinite(df["transaction_amount"])
            | (df["transaction_amount"] <= 0)
        ),

        "INVALID_STATUS": (
            ~df["status"].isin(VALID_STATUSES)
        ),

        "INVALID_TIMESTAMP": (
            df["transaction_timestamp"].isna()
        )
    }

    # Initialize rejection reason column
    df["rejection_reason"] = ""

    # Capture all rejection reasons
    for reason, condition in rules.items():
        df.loc[condition, "rejection_reason"] += reason + ";"

    df["rejection_reason"] = (
        df["rejection_reason"].str.rstrip(";")
    )

    # Separate invalid and valid records
    rejected_df = df[
        df["rejection_reason"] != ""
    ].copy()

    clean_df = df[
        df["rejection_reason"] == ""
    ].copy()

    # Remove rejection reason from clean records
    # Updated to avoid Pandas inplace deprecation warning
    clean_df = clean_df.drop(
        columns=["rejection_reason"]
    )

    # Add processing metadata
    processing_time = datetime.now(
        timezone.utc
    ).isoformat()

    clean_df["processed_timestamp"] = processing_time
    rejected_df["quarantine_timestamp"] = processing_time

    # Create output directories
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    QUARANTINE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save clean transactions
    clean_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Save rejected transactions
    rejected_df.to_csv(
        QUARANTINE_FILE,
        index=False
    )

    # Calculate data quality score
    quality_score = round(
        len(clean_df) / total_records * 100,
        2
    ) if total_records else 0.0

    # Verify that no records were lost
    reconciliation_passed = (
        total_records == len(clean_df) + len(rejected_df)
    )

    # Generate Silver quality report
    report = {
        "layer": "silver",

        "total_input_records": int(total_records),

        "clean_records": int(len(clean_df)),

        "rejected_records": int(len(rejected_df)),

        "data_quality_score": quality_score,

        "quality_issues": {
            reason: int(condition.sum())
            for reason, condition in rules.items()
        },

        "high_value_transactions_retained": int(
            (clean_df["transaction_amount"] >= 50000).sum()
        ),

        "reconciliation": (
            "PASS" if reconciliation_passed else "FAIL"
        ),

        "processing_timestamp": processing_time
    }

    # Save quality report as JSON
    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Display processing results
    print(f"Total input records : {total_records:,}")
    print(f"Clean records       : {len(clean_df):,}")
    print(f"Rejected records    : {len(rejected_df):,}")
    print(f"Data quality score  : {quality_score}%")

    print("-" * 55)
    print("DATA QUALITY ISSUES")
    print("-" * 55)

    for reason, count in report["quality_issues"].items():
        print(f"{reason:25}: {count:,}")

    print("-" * 55)

    print(
        "High-value records retained:",
        report["high_value_transactions_retained"]
    )

    print(f"Reconciliation: {report['reconciliation']}")
    print(f"Clean data saved: {OUTPUT_FILE}")
    print(f"Quarantine saved: {QUARANTINE_FILE}")
    print(f"Quality report saved: {REPORT_FILE}")

    print("SILVER PROCESSING COMPLETED")
    print("=" * 55)

    return report


if __name__ == "__main__":
    run_silver()
