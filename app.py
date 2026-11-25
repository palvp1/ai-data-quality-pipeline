
from pathlib import Path
import json
import sqlite3

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
DATABASE_FILE = (
    BASE_DIR / "data" / "gold" / "banking_analytics.db"
)

st.set_page_config(
    page_title="AI Data Quality Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# PROFESSIONAL LIGHT DASHBOARD STYLING
# =====================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #F5F8FC;
        color: #1E293B;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #EAF0FA;
        border-right: 1px solid #DCE5F2;
    }

    /* Headings */
    h1 {
        color: #172B4D !important;
        font-weight: 750 !important;
    }

    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px;
        min-height: 115px;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
        font-size: 30px;
        font-weight: 700;
    }

    /* Pastel KPI cards */
    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(1)
    [data-testid="stMetric"] {
        background-color: #E8F1FF;
        border-color: #C9DDFB;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(2)
    [data-testid="stMetric"] {
        background-color: #E5F6EF;
        border-color: #C5E8D8;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(3)
    [data-testid="stMetric"] {
        background-color: #FDECEF;
        border-color: #F4D2DB;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(4)
    [data-testid="stMetric"] {
        background-color: #FFF2DC;
        border-color: #F4DFB6;
    }

    /* Chart containers */
    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.03);
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #E8F1FF;
        color: #2563EB;
        border: 1px solid #C9DDFB;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #D5E5FF;
        color: #1D4ED8;
        border-color: #93B5F5;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }

    /* Dividers */
    hr {
        border-color: #E2E8F0 !important;
    }

</style>
""", unsafe_allow_html=True)


# =====================================================
# PASTEL CHART THEME
# =====================================================

import plotly.io as pio
import plotly.graph_objects as go

PASTEL_COLORS = [
    "#93B5F5",
    "#8DD8B8",
    "#B4A4E8",
    "#F6CF83",
    "#F2A5B5",
    "#85D5E5",
    "#D4B5E8",
    "#F5B895"
]

pio.templates["enterprise_light"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=PASTEL_COLORS,
        font=dict(
            family="Arial",
            size=13,
            color="#334155"
        ),
        title=dict(
            font=dict(
                size=18,
                color="#172B4D"
            )
        ),
        xaxis=dict(
            gridcolor="#E8EDF5",
            zerolinecolor="#E2E8F0"
        ),
        yaxis=dict(
            gridcolor="#E8EDF5",
            zerolinecolor="#E2E8F0"
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)"
        )
    )
)

pio.templates.default = "enterprise_light"

px.defaults.template = "enterprise_light"
px.defaults.color_discrete_sequence = PASTEL_COLORS


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_report(filename):

    path = REPORT_DIR / filename

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def query_database(query):

    with sqlite3.connect(DATABASE_FILE) as conn:
        return pd.read_sql_query(query, conn)


def show_table(df):

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )




# Load pipeline reports
gold = load_report("gold_report.json")
silver = load_report("silver_report.json")
anomaly = load_report("anomaly_report.json")
ai = load_report("ai_insights.json")
pipeline = load_report("pipeline_report.json")

# Sidebar
with st.sidebar:

    st.title("🤖 AI Data Quality")

    st.caption(
        "Banking Data Engineering Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Quality",
            "ML Anomalies",
            "Business Analytics",
            "GenAI Insights",
            "Pipeline Monitoring",
            "SQL Explorer"
        ]
    )

    st.divider()

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - Isolation Forest
    - Ollama
    - Llama 3.1 8B
    - SQLite
    - Streamlit
    - Plotly
    """)

    st.divider()

    if pipeline.get("status") == "SUCCESS":
        st.success("Pipeline: Operational")
    else:
        st.warning("Check pipeline status")

    if st.button("Refresh Dashboard", width="stretch"):
        st.rerun()

    st.caption("Portfolio Project | Synthetic Data")


# Validate required files
if not gold or not DATABASE_FILE.exists():

    st.error(
        "Pipeline outputs are missing. "
        "Run python run_pipeline.py first."
    )

    st.stop()


if pipeline.get("status") == "FAILED":
    st.warning(
        "The latest pipeline run failed. "
        "Some displayed outputs may be stale."
    )


# Main header
st.title("AI Data Quality Intelligence")

st.caption(
    "End-to-end banking data engineering, "
    "machine learning and generative AI monitoring"
)

st.divider()


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    st.header("Pipeline Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{gold['total_ingested_records']:,}"
    )

    c2.metric(
        "Data Quality Score",
        f"{gold['data_quality_score']:.2f}%"
    )

    c3.metric(
        "Quarantined Records",
        f"{gold['rejected_records']:,}"
    )

    c4.metric(
        "ML Anomalies",
        f"{gold['anomalous_records']:,}"
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Clean Transaction Value",
        f"${gold['total_transaction_amount']:,.2f}"
    )

    c6.metric(
        "Anomalous Value",
        f"${gold['anomalous_transaction_amount']:,.2f}"
    )

    c7.metric(
        "Pipeline Duration",
        f"{pipeline.get('total_duration_seconds', 0):.2f}s"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        distribution = pd.DataFrame({
            "Category": [
                "Normal",
                "Quarantined",
                "ML Anomalies"
            ],
            "Records": [
                gold["normal_records"],
                gold["rejected_records"],
                gold["anomalous_records"]
            ]
        })

        fig = px.pie(
            distribution,
            names="Category",
            values="Records",
            hole=0.55,
            title="Transaction Distribution",
            color="Category",
            color_discrete_map={
                "Normal": "#22C55E",
                "Quarantined": "#EF4444",
                "ML Anomalies": "#F59E0B"
            }
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        values = pd.DataFrame({
            "Category": ["Normal", "Anomalous"],
            "Amount": [
                gold["normal_transaction_amount"],
                gold["anomalous_transaction_amount"]
            ]
        })

        fig = px.bar(
            values,
            x="Category",
            y="Amount",
            color="Category",
            title="Transaction Value Distribution",
            color_discrete_sequence=[
                "#22C55E", "#F59E0B"
            ]
        )

        st.plotly_chart(fig, width="stretch")

    st.subheader("Pipeline Architecture")

    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style=rounded];

        Source [label="Synthetic Data"];
        Bronze [label="Bronze\\nIngestion"];
        Silver [label="Silver\\nData Quality"];
        ML [label="Isolation Forest\\nAnomaly Detection"];
        AI [label="Llama 3.1\\nGenAI Insights"];
        Gold [label="Gold\\nAnalytics"];
        DB [label="SQLite"];
        UI [label="Streamlit"];
        Quarantine [label="Quarantine"];

        Source -> Bronze;
        Bronze -> Silver;
        Silver -> ML;
        Silver -> Quarantine;
        ML -> AI;
        ML -> Gold;
        Gold -> DB;
        DB -> UI;
        AI -> UI;
    }
    """)

    st.subheader("Reconciliation Status")

    checks = {
        "Record Reconciliation":
            gold["record_reconciliation"],

        "Anomaly Reconciliation":
            gold["anomaly_reconciliation"],

        "Amount Reconciliation":
            gold["amount_reconciliation"],

        "Database Reconciliation":
            gold["database_reconciliation"]
    }

    columns = st.columns(4)

    for column, (name, result) in zip(
        columns, checks.items()
    ):

        with column:

            if result == "PASS":
                st.success(f"{name}: PASS")
            else:
                st.error(f"{name}: FAIL")


# =====================================================
# PAGE 2: DATA QUALITY
# =====================================================

elif page == "Data Quality":

    st.header("Data Quality Monitoring")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Quality Score",
        f"{gold['data_quality_score']:.2f}%"
    )

    c2.metric(
        "Clean Records",
        f"{gold['clean_records']:,}"
    )

    c3.metric(
        "Rejected Records",
        f"{gold['rejected_records']:,}"
    )

    quality = query_database(
        "SELECT * FROM quality_summary"
    )

    quality = quality.sort_values(
        "affected_records",
        ascending=True
    )

    fig = px.bar(
        quality,
        x="affected_records",
        y="quality_issue",
        orientation="h",
        color="affected_records",
        color_continuous_scale="Reds",
        title="Data Quality Issues by Category",
        labels={
            "affected_records": "Affected Records",
            "quality_issue": "Quality Issue"
        }
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Data Quality Summary")

    show_table(quality)

    st.subheader("Quarantined Transactions")

    rejected = query_database(
        "SELECT * FROM quarantined_transactions"
    )

    search = st.text_input(
        "Search rejection reason"
    )

    if search and "rejection_reason" in rejected.columns:

        rejected = rejected[
            rejected["rejection_reason"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                regex=False,
                na=False
            )
        ]

    st.caption(
        f"Matching records: {len(rejected):,}"
    )

    show_table(rejected)

    st.download_button(
        "Download Quarantine Report",
        data=rejected.to_csv(index=False),
        file_name="quarantined_transactions.csv",
        mime="text/csv"
    )


# =====================================================
# PAGE 3: ML ANOMALIES
# =====================================================

elif page == "ML Anomalies":

    st.header("Machine Learning Anomaly Detection")

    st.caption(
        "Model: Isolation Forest | Unsupervised Learning"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Analyzed Records",
        f"{gold['clean_records']:,}"
    )

    c2.metric(
        "Detected Anomalies",
        f"{gold['anomalous_records']:,}"
    )

    c3.metric(
        "Anomaly Rate",
        f"{gold['anomaly_rate']:.2f}%"
    )

    c4.metric(
        "High-value Detection",
        f"{gold['high_value_detection_rate']:.1f}%"
    )

    st.info(
        "Anomalies are unusual transactions requiring "
        "investigation. They are not confirmed fraud."
    )

    anomalies = query_database(
        "SELECT * FROM anomalies "
        "ORDER BY transaction_amount DESC"
    )

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            anomalies,
            x="transaction_amount",
            nbins=30,
            title="Anomalous Amount Distribution",
            color_discrete_sequence=["#F59E0B"]
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        anomaly_types = (
            anomalies.groupby("transaction_type")
            .size()
            .reset_index(name="anomaly_count")
        )

        fig = px.bar(
            anomaly_types,
            x="transaction_type",
            y="anomaly_count",
            title="Anomalies by Transaction Type",
            color="transaction_type"
        )

        st.plotly_chart(fig, width="stretch")

    st.subheader("Anomaly Investigation")

    minimum_amount = st.number_input(
        "Minimum Transaction Amount ($)",
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    filtered = anomalies[
        anomalies["transaction_amount"] >= minimum_amount
    ]

    st.caption(
        f"Matching anomalies: {len(filtered):,}"
    )

    show_table(filtered)

    st.download_button(
        "Download Anomaly Report",
        data=filtered.to_csv(index=False),
        file_name="anomalous_transactions.csv",
        mime="text/csv"
    )


# =====================================================
# PAGE 4: BUSINESS ANALYTICS
# =====================================================

elif page == "Business Analytics":

    st.header("Banking Business Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Transaction Value",
        f"${gold['total_transaction_amount']:,.2f}"
    )

    c2.metric(
        "Normal Transaction Value",
        f"${gold['normal_transaction_amount']:,.2f}"
    )

    c3.metric(
        "Anomalous Transaction Value",
        f"${gold['anomalous_transaction_amount']:,.2f}"
    )

    status = query_database(
        "SELECT * FROM status_summary"
    )

    types = query_database(
        "SELECT * FROM type_summary"
    )

    channels = query_database(
        "SELECT * FROM channel_summary"
    )

    countries = query_database(
        "SELECT * FROM country_summary"
    )

    daily = query_database(
        "SELECT * FROM daily_summary "
        "ORDER BY transaction_date"
    )

    left, right = st.columns(2)

    with left:

        fig = px.bar(
            status,
            x="status",
            y="total_amount",
            color="status",
            title="Transaction Value by Status"
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        fig = px.pie(
            types,
            names="transaction_type",
            values="transaction_count",
            hole=0.4,
            title="Transaction Types"
        )

        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    with left:

        fig = px.bar(
            channels,
            x="channel",
            y="transaction_count",
            color="channel",
            title="Transactions by Channel"
        )

        st.plotly_chart(fig, width="stretch")

    with right:

        fig = px.bar(
            countries,
            x="country",
            y="total_amount",
            color="country",
            title="Transaction Value by Country"
        )

        st.plotly_chart(fig, width="stretch")

    fig = px.line(
        daily,
        x="transaction_date",
        y="transaction_count",
        title="Daily Transaction Volume",
        markers=True
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Daily Transaction Summary")

    show_table(daily)


# =====================================================
# PAGE 5: GENAI INSIGHTS
# =====================================================

elif page == "GenAI Insights":

    st.header("Generative AI Investigation Assistant")

    if not ai:

        st.warning(
            "AI report not found. "
            "Run python ai_assistant.py."
        )

    else:

        if not pipeline.get("genai_enabled", True):
            st.warning(
                "GenAI was skipped in the latest run. "
                "This report may describe an earlier dataset."
            )

        elif (
            pipeline.get("started_at")
            and ai.get("generated_at")
            and ai["generated_at"] < pipeline["started_at"]
        ):
            st.warning(
                "This AI report predates the latest pipeline run."
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "AI Model",
            ai.get("model", "Unknown")
        )

        c2.metric(
            "Execution Mode",
            ai.get("execution_mode", "LOCAL")
        )

        c3.metric(
            "Generation Time",
            f"{ai.get('execution_time_seconds', 0):.2f}s"
        )

        st.success(
            "AI investigation report available"
        )

        st.caption(
            f"Generated at: {ai.get('generated_at', '')}"
        )

        st.divider()

        st.subheader("AI-Generated Investigation Report")

        insights = ai.get(
            "ai_insights",
            "No insights available."
        )

        st.markdown(insights)

        st.divider()

        st.warning(
            "AI-generated recommendations require "
            "human verification before remediation."
        )

        st.download_button(
            "Download AI Investigation Report",
            data=insights,
            file_name="ai_investigation_report.txt",
            mime="text/plain"
        )

        st.download_button(
            "Download Structured AI Report",
            data=json.dumps(ai, indent=4),
            file_name="ai_insights.json",
            mime="application/json"
        )


# =====================================================
# PAGE 6: PIPELINE MONITORING
# =====================================================

elif page == "Pipeline Monitoring":

    st.header("Pipeline Execution Monitoring")

    if not pipeline:

        st.warning("Pipeline execution report not found.")

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Pipeline Status",
            pipeline.get("status", "UNKNOWN")
        )

        c2.metric(
            "Successful Stages",
            pipeline.get("successful_stages", 0)
        )

        c3.metric(
            "Failed Stages",
            pipeline.get("failed_stages", 0)
        )

        c4.metric(
            "Total Duration",
            f"{pipeline.get('total_duration_seconds', 0):.2f}s"
        )

        st.caption(
            f"Run ID: {pipeline.get('run_id', '')}"
        )

        st.caption(
            f"Started at: {pipeline.get('started_at', '')}"
        )

        stages = pd.DataFrame(
            pipeline.get("stages", [])
        )

        if not stages.empty:

            fig = px.bar(
                stages,
                x="duration_seconds",
                y="stage",
                orientation="h",
                color="status",
                title="Stage Execution Performance",
                text_auto=".2f",
                color_discrete_map={
                    "SUCCESS": "#22C55E",
                    "FAILED": "#EF4444",
                    "SKIPPED": "#94A3B8",
                    "NOT_RUN": "#94A3B8"
                }
            )

            st.plotly_chart(fig, width="stretch")

            st.subheader("Execution Details")

            show_table(stages)

        st.subheader("Pipeline Execution Log")

        with st.expander("View Complete JSON Report"):

            st.json(pipeline)

        st.download_button(
            "Download Pipeline Execution Report",
            data=json.dumps(pipeline, indent=4),
            file_name="pipeline_report.json",
            mime="application/json"
        )


# =====================================================
# PAGE 7: SQL EXPLORER
# =====================================================

elif page == "SQL Explorer":

    st.header("SQL Data Warehouse Explorer")

    st.caption(
        "Execute read-only SQL queries "
        "against the SQLite analytics warehouse."
    )

    tables = query_database("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """)

    st.subheader("Available Database Tables")

    show_table(tables)

    examples = {
        "Top 20 Anomalous Transactions": """
SELECT transaction_id,
       transaction_amount,
       transaction_type,
       status
FROM anomalies
ORDER BY transaction_amount DESC
LIMIT 20;
""",

        "Transaction Status Summary": """
SELECT *
FROM status_summary;
""",

        "Data Quality Issues": """
SELECT *
FROM quality_summary
ORDER BY affected_records DESC;
""",

        "Transaction Count by Country": """
SELECT country,
       COUNT(*) AS total_transactions,
       ROUND(SUM(transaction_amount), 2) AS total_amount
FROM transactions
GROUP BY country
ORDER BY total_amount DESC;
"""
    }

    selected = st.selectbox(
        "Choose an example query",
        list(examples.keys())
    )

    sql = st.text_area(
        "SQL Query",
        value=examples[selected].strip(),
        height=220,
        key=f"sql_{selected}"
    )

    if st.button(
        "Execute SQL Query",
        type="primary"
    ):

        if not sql.strip().upper().startswith(
            ("SELECT", "WITH")
        ):

            st.error(
                "Only read-only SELECT queries are allowed."
            )

        else:

            try:

                database_uri = (
                    DATABASE_FILE.resolve().as_uri()
                    + "?mode=ro"
                )

                with sqlite3.connect(
                    database_uri,
                    uri=True
                ) as conn:

                    conn.execute("PRAGMA query_only = ON")

                    result = pd.read_sql_query(
                        sql,
                        conn
                    )

                st.success(
                    f"Query completed: {len(result):,} rows"
                )

                show_table(result)

                st.download_button(
                    "Download Query Results",
                    data=result.to_csv(index=False),
                    file_name="sql_query_results.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error(f"SQL execution error: {error}")


# Footer
st.divider()

st.caption(
    "AI Data Quality Intelligence | "
    "Data Engineering + Machine Learning + Generative AI | "
    "All banking data is synthetic."
)
