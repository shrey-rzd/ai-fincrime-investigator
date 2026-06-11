import matplotlib.pyplot as plt
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import datetime
import json
import pandas as pd
import re

# ======================
# ✅ SESSION STATE
# ======================
if "case_data" not in st.session_state:
    st.session_state.case_data = None

# ======================
# LOAD API KEY
# ======================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

client = OpenAI(
    api_key=api_key,
    http_client=httpx.Client(verify=False)
)

# ======================
# LOAD DATASET
# ======================
data = pd.read_csv("fraud_data.csv")

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(text):
    amount_match = re.search(r'\d+', text)
    amount = float(amount_match.group()) if amount_match else 0

    if "PAYMENT" in text:
        txn_type = "PAYMENT"
    elif "TRANSFER" in text:
        txn_type = "TRANSFER"
    elif "CASH_OUT" in text:
        txn_type = "CASH_OUT"
    else:
        txn_type = "OTHER"

    return amount, txn_type

data["amount"] = data["text"].apply(lambda x: extract_features(x)[0])
data["txn_type"] = data["text"].apply(lambda x: extract_features(x)[1])

# ======================
# RISK SCORING
# ======================
def calculate_risk(row):
    score = 0

    if row["amount"] > 200000:
        score += 6
    elif row["amount"] > 100000:
        score += 4
    elif row["amount"] > 50000:
        score += 2

    if row["txn_type"] == "CASH_OUT":
        score += 4
    elif row["txn_type"] == "TRANSFER":
        score += 3
    elif row["txn_type"] == "PAYMENT":
        score += 1

    return min(score, 10)

data["risk_score"] = data.apply(calculate_risk, axis=1)

# ✅ FILTER HIGH RISK
suspicious_data = data[data["risk_score"] >= 7]
suspicious_data = suspicious_data.sort_values(by="risk_score", ascending=False)

# ======================
# SAVE CASE
# ======================
def save_case(alert, findings, report):
    case = {
        "case_id": f"CASE_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": str(datetime.datetime.now()),
        "alert": alert,
        "findings": findings,
        "report": report
    }

    with open("case_history.json", "a") as f:
        f.write(json.dumps(case) + "\n")

    return case

# ======================
# ✅ SAVE EVALUATION (FIXED)
# ======================
def save_evaluation(case_id, risk_score, decision):

    record = {
        "case_id": str(case_id),
        "risk_score": float(risk_score),
        "decision": str(decision),
        "timestamp": str(datetime.datetime.now())
    }

    try:
        with open("evaluation_log.json", "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print("Error saving evaluation:", e)

# ======================
# UI HEADER
# ======================
st.title("🚨 Financial Crime Investigation Platform")

# ======================
# SIDEBAR
# ======================
st.sidebar.subheader("High Risk Cases")
st.sidebar.write(suspicious_data.head(10))

if len(suspicious_data) > 0:
    selected_index = st.sidebar.number_input(
        "Select Case",
        min_value=0,
        max_value=len(suspicious_data)-1,
        value=0
    )

    if st.sidebar.button("Run Investigation"):

        row = suspicious_data.iloc[selected_index]

        text = row["text"]
        amount = row["amount"]
        txn_type = row["txn_type"]
        risk_score = row["risk_score"]

        alert = {
            "customer_id": f"C{selected_index}",
            "amount": amount,
            "country": "India",
            "alert_type": txn_type
        }

        findings = []

        if amount > 100000:
            findings.append("High transaction amount")

        if txn_type in ["TRANSFER", "CASH_OUT"]:
            findings.append("High-risk transaction type")

        # AI REPORT
        report_prompt = f"""
        Investigate fraud:

        Alert: {alert}
        Findings: {findings}

        Provide Risk Level, Reasons, Recommendation
        """

        report = client.responses.create(
            model="gpt-4.1-mini",
            input=report_prompt
        ).output_text

        case = save_case(alert, findings, report)

        # ✅ STORE SESSION DATA
        st.session_state.case_data = {
            "text": text,
            "alert": alert,
            "findings": findings,
            "report": report,
            "risk_score": risk_score,
            "case_id": case["case_id"]
        }

# ======================
# ✅ MAIN DISPLAY
# ======================
if st.session_state.case_data:

    case_data = st.session_state.case_data

    # 🔷 TOP BAR
    st.markdown("### 🧾 Case Overview")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Case ID", case_data["case_id"])
    c2.metric("Amount", f"₹ {case_data['alert']['amount']}")
    c3.metric("Type", case_data["alert"]["alert_type"])
    c4.metric("Risk Score", f"{case_data['risk_score']}/10")

    st.divider()

    # 🔷 TRANSACTION
    st.subheader("📥 Transaction")
    st.info(case_data["text"])

    # 🔷 ALERT
    with st.expander("📊 Alert Details"):
        st.json(case_data["alert"])

    # 🔷 FINDINGS
    st.subheader("⚠️ Key Findings")
    for f in case_data["findings"]:
        st.markdown(
            f"<span style='background:#fee2e2;color:#991b1b;padding:6px 10px;border-radius:15px;margin-right:6px;'>{f}</span>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 🔷 REPORT
    st.subheader("📄 Investigation Report")
    st.success(case_data["report"])

    # 🔷 QA AGENT
    st.subheader("💬 Ask Investigator")
    question = st.text_input("Ask a question")

    if st.button("Ask") and question:
        answer = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Case: {case_data} \n Question: {question}"
        )
        st.info(answer.output_text)

    # 🔷 DECISION
    st.subheader("👨‍💼 Analyst Decision")
    decision = st.radio("", ["Approve ✅", "Reject ❌", "Escalate 🔴"])

    if st.button("Submit Decision"):
        st.success(f"{decision} recorded ✅")

        with open("decisions_log.txt", "a") as f:
            f.write(f"{case_data['case_id']} - {decision}\n")

        save_evaluation(
            case_data["case_id"],
            case_data["risk_score"],
            decision
        )

    # 🔷 RISK STATUS
    st.subheader("📊 Risk Status")

    if case_data["risk_score"] >= 7:
        st.error("🔴 HIGH RISK")
    else:
        st.success("🟢 LOW RISK")

# ======================
# 📊 DASHBOARD ANALYTICS
# ======================
st.subheader("📊 System Analytics")

if len(data) > 0:

    col1, col2 = st.columns(2)

    # ✅ Risk Score Distribution
    with col1:
        st.markdown("### Risk Score Distribution")

        fig, ax = plt.subplots()
        data["risk_score"].hist(ax=ax)
        ax.set_title("Risk Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    # ✅ Transaction Type Distribution
    with col2:
        st.markdown("### Transaction Type Distribution")

        fig2, ax2 = plt.subplots()
        data["txn_type"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Transaction Types")

        st.pyplot(fig2)

    # ✅ Key Metrics
    st.markdown("### Key Metrics")

    total = len(data)
    high_risk = len(data[data["risk_score"] >= 7])

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Transactions", total)
    c2.metric("High Risk Cases", high_risk)
    c3.metric("High Risk %", f"{round(high_risk/total*100,2)}%")