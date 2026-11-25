
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys
import time
import json
import argparse
import uuid

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

REPORT_FILE = REPORT_DIR / "pipeline_report.json"

PIPELINE_STAGES = [
    ("Data Generation", "generate_data.py"),
    ("Bronze Ingestion", "bronze.py"),
    ("Silver Data Quality", "silver.py"),
    ("ML Anomaly Detection", "anomaly_detection.py"),
    ("GenAI Investigation", "ai_assistant.py"),
    ("Gold Business Analytics", "gold.py")
]


def execute_stage(stage_name, script_name):

    print("\n" + "=" * 65)
    print(f"STARTING STAGE: {stage_name}")
    print("=" * 65, flush=True)

    start_time = time.perf_counter()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / script_name)
            ],
            cwd=BASE_DIR,
            check=True,
            timeout=900
        )

        status = "SUCCESS"
        error_message = None

    except subprocess.CalledProcessError as error:

        status = "FAILED"
        error_message = (
            f"Process exited with code "
            f"{error.returncode}"
        )

    except subprocess.TimeoutExpired:

        status = "FAILED"
        error_message = "Stage execution timed out."

    except Exception as error:

        status = "FAILED"
        error_message = str(error)

    duration = round(
        time.perf_counter() - start_time,
        2
    )

    result = {
        "stage": stage_name,
        "script": script_name,
        "status": status,
        "duration_seconds": duration,
        "started_at": started_at,
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "error": error_message
    }

    print("-" * 65)
    print(f"Stage: {stage_name}")
    print(f"Status: {status}")
    print(f"Duration: {duration} seconds")

    if error_message:
        print(f"Error: {error_message}")

    return result


def run_pipeline(skip_ai=False):

    pipeline_start = time.perf_counter()

    run_id = str(uuid.uuid4())

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("AI-POWERED DATA QUALITY PIPELINE")
    print("=" * 65)
    print(f"Pipeline run ID: {run_id}")
    print(f"Started at: {started_at}")
    print(f"Python: {sys.executable}")

    results = []
    pipeline_status = "SUCCESS"

    for stage_name, script_name in PIPELINE_STAGES:

        if pipeline_status == "FAILED":

            results.append({
                "stage": stage_name,
                "script": script_name,
                "status": "NOT_RUN",
                "duration_seconds": 0,
                "error": "Previous stage failed."
            })

            continue

        if skip_ai and script_name == "ai_assistant.py":

            results.append({
                "stage": stage_name,
                "script": script_name,
                "status": "SKIPPED",
                "duration_seconds": 0,
                "error": None
            })

            print("\nGenAI stage skipped.")

            continue

        result = execute_stage(
            stage_name,
            script_name
        )

        results.append(result)

        if result["status"] == "FAILED":

            pipeline_status = "FAILED"

            print("\nPIPELINE FAILURE DETECTED")
            print("Dependent stages will not execute.")

    total_duration = round(
        time.perf_counter() - pipeline_start,
        2
    )

    successful_stages = sum(
        stage["status"] == "SUCCESS"
        for stage in results
    )

    failed_stages = sum(
        stage["status"] == "FAILED"
        for stage in results
    )

    skipped_stages = sum(
        stage["status"] == "SKIPPED"
        for stage in results
    )

    not_run_stages = sum(
        stage["status"] == "NOT_RUN"
        for stage in results
    )

    report = {
        "pipeline_name": "AI Data Quality Pipeline",
        "run_id": run_id,
        "status": pipeline_status,
        "started_at": started_at,
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_duration_seconds": total_duration,
        "total_stages": len(PIPELINE_STAGES),
        "successful_stages": successful_stages,
        "failed_stages": failed_stages,
        "skipped_stages": skipped_stages,
        "not_run_stages": not_run_stages,
        "genai_enabled": not skip_ai,
        "stages": results
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 65)

    for stage in results:

        print(
            f"{stage['stage']:25} | "
            f"{stage['status']:10} | "
            f"{stage['duration_seconds']:8.2f}s"
        )

    print("-" * 65)

    print(f"Total stages      : {len(PIPELINE_STAGES)}")
    print(f"Successful stages : {successful_stages}")
    print(f"Failed stages     : {failed_stages}")
    print(f"Skipped stages    : {skipped_stages}")
    print(f"Not run stages    : {not_run_stages}")
    print(f"Total duration    : {total_duration} seconds")
    print(f"Pipeline status   : {pipeline_status}")
    print(f"Report saved      : {REPORT_FILE}")

    print("=" * 65)

    return report


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AI Data Quality Pipeline"
    )

    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip the GenAI investigation stage"
    )

    args = parser.parse_args()

    report = run_pipeline(
        skip_ai=args.skip_ai
    )

    sys.exit(
        0 if report["status"] == "SUCCESS" else 1
    )
