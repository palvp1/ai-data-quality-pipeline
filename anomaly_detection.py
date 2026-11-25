
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "silver" / "clean_transactions.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "silver" / "scored_transactions.csv"
)

ANOMALY_FILE = (
    BASE_DIR / "data" / "silver" / "anomalous_transactions.csv"
)

MODEL_FILE = (
    BASE_DIR / "models" / "isolation_forest.joblib"
)

REPORT_FILE = (
    BASE_DIR / "reports" / "anomaly_report.json"
)

CONTAMINATION = 0.01
RANDOM_SEED = 42


def run_anomaly_detection():

    print("=" * 55)
    print("AI ENGINE - ML ANOMALY DETECTION")
    print("=" * 55)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Silver data not found. Run silver.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("No clean transactions available.")

    # Feature engineering
    features = pd.DataFrame({
        "log_transaction_amount": np.log1p(
            df["transaction_amount"]
        )
    })

    print(f"Records loaded: {len(df):,}")
    print("Feature engineering: COMPLETED")

    # Initialize Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Train model and generate predictions
    predictions = model.fit_predict(features)

    # -1 = anomaly, 1 = normal
    df["is_anomaly"] = (
        predictions == -1
    ).astype(int)

    # Higher score indicates greater abnormality
    df["anomaly_score"] = -model.decision_function(
        features
    )

    df["anomaly_status"] = np.where(
        df["is_anomaly"] == 1,
        "ANOMALY",
        "NORMAL"
    )

    # Separate anomalous transactions
    anomalies = df[
        df["is_anomaly"] == 1
    ].copy()

    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False
    )

    # Create directories
    for path in [
        OUTPUT_FILE,
        ANOMALY_FILE,
        MODEL_FILE,
        REPORT_FILE
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # Save model and scored transactions
    joblib.dump(model, MODEL_FILE)

    df.to_csv(OUTPUT_FILE, index=False)

    anomalies.to_csv(
        ANOMALY_FILE,
        index=False
    )

    # High-value transaction analysis
    high_value_mask = (
        df["transaction_amount"] >= 50000
    )

    high_value_count = int(
        high_value_mask.sum()
    )

    high_value_detected = int(
        (
            high_value_mask
            & (df["is_anomaly"] == 1)
        ).sum()
    )

    detection_rate = round(
        high_value_detected / high_value_count * 100,
        2
    ) if high_value_count else 0.0

    # Generate anomaly report
    report = {
        "algorithm": "Isolation Forest",
        "total_records": int(len(df)),
        "normal_records": int(
            (df["is_anomaly"] == 0).sum()
        ),
        "anomalous_records": int(len(anomalies)),
        "anomaly_rate": round(
            len(anomalies) / len(df) * 100,
            2
        ),
        "high_value_transactions": high_value_count,
        "high_value_anomalies_detected": high_value_detected,
        "high_value_detection_rate": detection_rate,
        "total_anomalous_amount": round(
            float(anomalies["transaction_amount"].sum()),
            2
        ),
        "model_parameters": {
            "n_estimators": 100,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_SEED
        },
        "features": list(features.columns),
        "execution_timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Display results
    print("Model training: COMPLETED")
    print("-" * 55)

    print(f"Total records analyzed : {len(df):,}")
    print(f"Normal records         : {report['normal_records']:,}")
    print(f"Anomalous records      : {len(anomalies):,}")
    print(f"Anomaly rate           : {report['anomaly_rate']}%")

    print("-" * 55)

    print(f"High-value transactions: {high_value_count:,}")
    print(f"High-value detected    : {high_value_detected:,}")
    print(f"High-value detection   : {detection_rate}%")

    print(
        f"Total anomalous amount : "
        f"${report['total_anomalous_amount']:,.2f}"
    )

    print("-" * 55)
    print(f"Model saved: {MODEL_FILE}")
    print(f"Scored data saved: {OUTPUT_FILE}")
    print(f"Anomalies saved: {ANOMALY_FILE}")
    print(f"Report saved: {REPORT_FILE}")
    print("AI ANOMALY DETECTION COMPLETED")
    print("=" * 55)

    return report


if __name__ == "__main__":
    run_anomaly_detection()
