
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import requests

BASE_DIR = Path(__file__).resolve().parent

REPORT_DIR = BASE_DIR / "reports"

BRONZE_REPORT = REPORT_DIR / "bronze_report.json"
SILVER_REPORT = REPORT_DIR / "silver_report.json"
ANOMALY_REPORT = REPORT_DIR / "anomaly_report.json"

OUTPUT_FILE = REPORT_DIR / "ai_insights.json"
TEXT_REPORT = REPORT_DIR / "ai_insights.txt"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def load_report(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required report not found: {file_path.name}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_prompt(bronze, silver, anomaly):

    pipeline_metrics = {
        "bronze_layer": bronze,
        "silver_layer": silver,
        "anomaly_detection": anomaly
    }

    metrics_json = json.dumps(
        pipeline_metrics,
        indent=4
    )

    prompt = f"""
You are a senior data engineer specializing in
banking data quality and pipeline monitoring.

Analyze the following synthetic banking data
pipeline metrics.

PIPELINE METRICS:

{metrics_json}

Generate a professional investigation report
with the following sections:

1. EXECUTIVE SUMMARY
Summarize the pipeline execution results.

2. PIPELINE HEALTH ASSESSMENT
Assess the data quality score, rejected records,
reconciliation status and anomaly rate.

3. DATA QUALITY INVESTIGATION
Explain the significance of missing customer IDs,
duplicate transactions, invalid amounts,
invalid statuses and invalid timestamps.

4. POTENTIAL ROOT CAUSES
Suggest possible causes for the observed issues.
Clearly identify them as hypotheses, not
confirmed root causes.

5. ML ANOMALY ANALYSIS
Explain the anomaly detection results.
Do not classify anomalies as confirmed fraud.
Do not describe high-value detection rate
as overall model accuracy.

6. REMEDIATION RECOMMENDATIONS
Provide practical actions that a data engineer
can take to investigate and resolve the issues.

7. PREVENTIVE MEASURES
Recommend automated validation, monitoring,
alerting and regression testing improvements.

8. PRIORITY ACTION PLAN
Provide the three most important next actions.

IMPORTANT RULES:

- Use only the supplied pipeline metrics.
- Do not invent numerical results.
- Distinguish facts from hypotheses.
- All transaction data is synthetic.
- Do not recommend deleting financial records.
- Recommend human review before remediation.
- Keep the report concise and professional.
"""

    return prompt


def generate_ai_insights(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1500,
            "num_ctx": 4096
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(10, 600)
        )

        response.raise_for_status()

        result = response.json()

        insights = result.get("response", "").strip()

        if not insights:
            raise ValueError(
                "Ollama returned an empty response."
            )

        return insights

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Unable to connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "The AI model exceeded the request timeout."
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Ollama API request failed: {error}"
        )


def run_ai_assistant():

    start_time = time.perf_counter()

    print("=" * 55)
    print("GENAI DATA QUALITY INVESTIGATION ASSISTANT")
    print("=" * 55)

    # Load pipeline reports
    bronze = load_report(BRONZE_REPORT)
    silver = load_report(SILVER_REPORT)
    anomaly = load_report(ANOMALY_REPORT)

    print("Bronze report loaded  : SUCCESS")
    print("Silver report loaded  : SUCCESS")
    print("Anomaly report loaded : SUCCESS")

    # Construct the LLM prompt
    prompt = build_prompt(
        bronze,
        silver,
        anomaly
    )

    print("Prompt engineering    : COMPLETED")
    print(f"AI model              : {MODEL_NAME}")
    print(f"API endpoint          : {OLLAMA_URL}")

    print("-" * 55)
    print("Generating AI investigation report...")
    print("Please wait. This may take several minutes.")

    # Generate insights using Ollama
    insights = generate_ai_insights(prompt)

    execution_time = round(
        time.perf_counter() - start_time,
        2
    )

    # Prepare structured report
    report = {
        "status": "SUCCESS",
        "model": MODEL_NAME,
        "provider": "Ollama",
        "execution_mode": "LOCAL",
        "data_quality_score": silver[
            "data_quality_score"
        ],
        "total_records": bronze[
            "total_records"
        ],
        "clean_records": silver[
            "clean_records"
        ],
        "rejected_records": silver[
            "rejected_records"
        ],
        "anomalous_records": anomaly[
            "anomalous_records"
        ],
        "reconciliation": silver[
            "reconciliation"
        ],
        "ai_insights": insights,
        "execution_time_seconds": execution_time,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save structured JSON report
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save human-readable report
    with open(
        TEXT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(insights)

    # Display AI-generated insights
    print("=" * 55)
    print("AI GENERATED INVESTIGATION REPORT")
    print("=" * 55)

    print(insights)

    print("-" * 55)
    print(f"AI model: {MODEL_NAME}")
    print(f"Execution time: {execution_time} seconds")
    print(f"JSON report saved: {OUTPUT_FILE}")
    print(f"Text report saved: {TEXT_REPORT}")
    print("GENAI ANALYSIS COMPLETED")
    print("=" * 55)

    return report


if __name__ == "__main__":
    run_ai_assistant()
