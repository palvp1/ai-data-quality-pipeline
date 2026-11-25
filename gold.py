
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
GOLD_DIR = DATA_DIR / "gold"
REPORT_DIR = BASE_DIR / "reports"

SCORED_FILE = (
    DATA_DIR / "silver" / "scored_transactions.csv"
)

QUARANTINE_FILE = (
    DATA_DIR / "quarantine" / "rejected_transactions.csv"
)

DATABASE_FILE = GOLD_DIR / "banking_analytics.db"

REPORT_FILE = REPORT_DIR / "gold_report.json"


def load_report(filename):

    with open(
        REPORT_DIR / filename,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def run_gold():

    print("=" * 55)
    print("GOLD LAYER - BUSINESS ANALYTICS")
    print("=" * 55)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed transactions
    df = pd.read_csv(SCORED_FILE)

    rejected = pd.read_csv(QUARANTINE_FILE)

    bronze = load_report("bronze_report.json")
    silver = load_report("silver_report.json")
    anomaly = load_report("anomaly_report.json")

    print(f"Clean records loaded: {len(df):,}")
    print(f"Quarantined records : {len(rejected):,}")

    # Separate normal and anomalous transactions
    anomalies = df[df["is_anomaly"] == 1].copy()

    normal = df[df["is_anomaly"] == 0].copy()

    # 1. Transaction status analytics
    status_summary = (
        df.groupby("status")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("transaction_amount", "sum"),
            average_amount=("transaction_amount", "mean"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    # 2. Transaction type analytics
    type_summary = (
        df.groupby("transaction_type")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("transaction_amount", "sum"),
            average_amount=("transaction_amount", "mean"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    # 3. Channel analytics
    channel_summary = (
        df.groupby("channel")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("transaction_amount", "sum"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    # 4. Country analytics
    country_summary = (
        df.groupby("country")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("transaction_amount", "sum"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    # 5. Daily transaction analytics
    df["transaction_date"] = pd.to_datetime(
        df["transaction_timestamp"]
    ).dt.strftime("%Y-%m-%d")

    daily_summary = (
        df.groupby("transaction_date")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("transaction_amount", "sum"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )

    # 6. Data quality analytics
    quality_summary = pd.DataFrame([
        {
            "quality_issue": reason,
            "affected_records": count
        }
        for reason, count
        in silver["quality_issues"].items()
    ])

    # 7. Pipeline reconciliation
    bronze_count = bronze["total_records"]

    clean_count = len(df)
    rejected_count = len(rejected)

    normal_count = len(normal)
    anomaly_count = len(anomalies)

    total_amount = round(
        float(df["transaction_amount"].sum()), 2
    )

    normal_amount = round(
        float(normal["transaction_amount"].sum()), 2
    )

    anomaly_amount = round(
        float(anomalies["transaction_amount"].sum()), 2
    )

    record_reconciliation = (
        bronze_count == clean_count + rejected_count
    )

    anomaly_reconciliation = (
        clean_count == normal_count + anomaly_count
    )

    amount_reconciliation = (
        abs(
            total_amount
            - normal_amount
            - anomaly_amount
        ) < 0.02
    )

    reconciliation_passed = all([
        record_reconciliation,
        anomaly_reconciliation,
        amount_reconciliation
    ])

    if not reconciliation_passed:
        raise ValueError("Gold reconciliation failed.")

    # 8. Pipeline KPI summary
    kpi = {
        "total_ingested_records": int(bronze_count),
        "clean_records": int(clean_count),
        "rejected_records": int(rejected_count),
        "normal_records": int(normal_count),
        "anomalous_records": int(anomaly_count),
        "data_quality_score": silver["data_quality_score"],
        "anomaly_rate": anomaly["anomaly_rate"],
        "high_value_detection_rate": anomaly[
            "high_value_detection_rate"
        ],
        "total_transaction_amount": total_amount,
        "normal_transaction_amount": normal_amount,
        "anomalous_transaction_amount": anomaly_amount,
        "record_reconciliation": (
            "PASS" if record_reconciliation else "FAIL"
        ),
        "anomaly_reconciliation": (
            "PASS" if anomaly_reconciliation else "FAIL"
        ),
        "amount_reconciliation": (
            "PASS" if amount_reconciliation else "FAIL"
        ),
        "overall_reconciliation": (
            "PASS" if reconciliation_passed else "FAIL"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    kpi_summary = pd.DataFrame([kpi])

    # Round financial analytics
    summaries = [
        status_summary,
        type_summary,
        channel_summary,
        country_summary,
        daily_summary
    ]

    for summary in summaries:

        for column in [
            "total_amount",
            "average_amount"
        ]:

            if column in summary.columns:
                summary[column] = (
                    summary[column].round(2)
                )

    # Save Gold CSV files
    tables = {
        "status_summary": status_summary,
        "type_summary": type_summary,
        "channel_summary": channel_summary,
        "country_summary": country_summary,
        "daily_summary": daily_summary,
        "quality_summary": quality_summary,
        "pipeline_kpis": kpi_summary
    }

    for name, table in tables.items():

        table.to_csv(
            GOLD_DIR / f"{name}.csv",
            index=False
        )

    print("Gold CSV generation: SUCCESS")

    # Load data into SQLite
    with sqlite3.connect(DATABASE_FILE) as conn:

        df.to_sql(
            "transactions",
            conn,
            if_exists="replace",
            index=False
        )

        anomalies.to_sql(
            "anomalies",
            conn,
            if_exists="replace",
            index=False
        )

        rejected.to_sql(
            "quarantined_transactions",
            conn,
            if_exists="replace",
            index=False
        )

        for name, table in tables.items():

            table.to_sql(
                name,
                conn,
                if_exists="replace",
                index=False
            )

        # Create database indexes
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_transaction_id
            ON transactions(transaction_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_anomaly_status
            ON transactions(is_anomaly)
        """)

        # Verify database record counts
        db_count = conn.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

        db_anomaly_count = conn.execute(
            "SELECT COUNT(*) FROM anomalies"
        ).fetchone()[0]

        db_rejected_count = conn.execute(
            "SELECT COUNT(*) FROM quarantined_transactions"
        ).fetchone()[0]

        database_reconciliation = (
            db_count == clean_count
            and db_anomaly_count == anomaly_count
            and db_rejected_count == rejected_count
        )

        if not database_reconciliation:
            raise ValueError(
                "Database reconciliation failed."
            )

    # Save final Gold report
    kpi["database_records"] = int(db_count)

    kpi["database_reconciliation"] = (
        "PASS" if database_reconciliation else "FAIL"
    )

    kpi["database_tables"] = 10

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(kpi, file, indent=4)

    # Display results
    print("SQLite database loading: SUCCESS")

    print("-" * 55)
    print("GOLD BUSINESS ANALYTICS")
    print("-" * 55)

    print(f"Total ingested records : {bronze_count:,}")
    print(f"Clean records          : {clean_count:,}")
    print(f"Quarantined records    : {rejected_count:,}")
    print(f"Normal records         : {normal_count:,}")
    print(f"Anomalous records      : {anomaly_count:,}")

    print(
        f"Data quality score     : "
        f"{silver['data_quality_score']}%"
    )

    print("-" * 55)

    print(f"Total transaction value: ${total_amount:,.2f}")
    print(f"Normal transaction value: ${normal_amount:,.2f}")
    print(f"Anomalous value        : ${anomaly_amount:,.2f}")

    print("-" * 55)
    print("RECONCILIATION RESULTS")
    print("-" * 55)

    print(
        "Record reconciliation  :",
        kpi["record_reconciliation"]
    )

    print(
        "Anomaly reconciliation :",
        kpi["anomaly_reconciliation"]
    )

    print(
        "Amount reconciliation  :",
        kpi["amount_reconciliation"]
    )

    print(
        "Database reconciliation:",
        kpi["database_reconciliation"]
    )

    print(f"Database records       : {db_count:,}")
    print("Database tables        : 10")

    print("-" * 55)

    print(f"Database saved: {DATABASE_FILE}")
    print(f"Gold report saved: {REPORT_FILE}")

    print("GOLD PROCESSING COMPLETED")
    print("=" * 55)

    return kpi


if __name__ == "__main__":
    run_gold()
