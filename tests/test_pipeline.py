from pathlib import Path
import sqlite3
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parents[1]

BRONZE = BASE_DIR / "data/bronze/bronze_transactions.csv"
SILVER = BASE_DIR / "data/silver/clean_transactions.csv"
QUARANTINE = BASE_DIR / "data/quarantine/rejected_transactions.csv"
ANOMALIES = BASE_DIR / "data/silver/anomalous_transactions.csv"
DATABASE = BASE_DIR / "data/gold/banking_analytics.db"


@pytest.fixture(scope="module")
def bronze():
    return pd.read_csv(BRONZE)


@pytest.fixture(scope="module")
def silver():
    return pd.read_csv(SILVER)


@pytest.fixture(scope="module")
def quarantine():
    return pd.read_csv(QUARANTINE)


@pytest.fixture(scope="module")
def anomalies():
    return pd.read_csv(ANOMALIES)


# 1. Bronze ingestion tests

def test_bronze_file_exists():
    assert BRONZE.exists()


def test_bronze_record_count(bronze):
    assert len(bronze) == 50500


def test_bronze_duplicate_count(bronze):
    assert bronze["transaction_id"].duplicated().sum() == 500


# 2. Silver data quality tests

def test_silver_file_exists():
    assert SILVER.exists()


def test_clean_record_count(silver):
    assert len(silver) == 48800


def test_no_duplicate_transactions(silver):
    assert silver["transaction_id"].duplicated().sum() == 0


def test_no_missing_transaction_ids(silver):
    assert silver["transaction_id"].notna().all()


def test_no_missing_customer_ids(silver):
    assert silver["customer_id"].notna().all()


def test_valid_transaction_amounts(silver):
    assert (silver["transaction_amount"] > 0).all()


def test_valid_transaction_status(silver):
    valid_statuses = ["SUCCESS", "FAILED", "PENDING"]
    assert silver["status"].isin(valid_statuses).all()


# 3. Quarantine and reconciliation tests

def test_quarantine_file_exists():
    assert QUARANTINE.exists()


def test_quarantine_record_count(quarantine):
    assert len(quarantine) == 1700


def test_record_reconciliation(bronze, silver, quarantine):
    assert len(bronze) == len(silver) + len(quarantine)


def test_data_quality_score(bronze, silver):
    score = round(len(silver) / len(bronze) * 100, 2)
    assert score == 96.63


# 4. Machine learning tests

def test_anomaly_file_exists():
    assert ANOMALIES.exists()


def test_anomaly_count(anomalies):
    assert len(anomalies) == 484


def test_anomalies_belong_to_clean_data(silver, anomalies):
    assert anomalies["transaction_id"].isin(
        silver["transaction_id"]
    ).all()


def test_anomaly_rate(silver, anomalies):
    rate = round(len(anomalies) / len(silver) * 100, 2)
    assert rate == 0.99


# 5. Database validation tests

def test_database_exists():
    assert DATABASE.exists()


def test_database_record_count(silver):
    with sqlite3.connect(
        DATABASE.as_uri() + "?mode=ro", uri=True
    ) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

    assert count == len(silver)


def test_database_anomaly_count(anomalies):
    with sqlite3.connect(
        DATABASE.as_uri() + "?mode=ro", uri=True
    ) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM anomalies"
        ).fetchone()[0]

    assert count == len(anomalies)


def test_database_table_count():
    with sqlite3.connect(
        DATABASE.as_uri() + "?mode=ro", uri=True
    ) as conn:
        count = conn.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
        """).fetchone()[0]

    assert count == 10


def test_database_amount_reconciliation(silver):
    with sqlite3.connect(
        DATABASE.as_uri() + "?mode=ro", uri=True
    ) as conn:
        total = conn.execute("""
            SELECT SUM(transaction_amount)
            FROM transactions
        """).fetchone()[0]

    assert total == pytest.approx(
        silver["transaction_amount"].sum(),
        abs=0.01,
        rel=0
    )