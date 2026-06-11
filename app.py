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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ======================
# 🔐 API KEY
# ======================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))

# ======================
# 🧾 UI
# ======================
st.title("🚨 Financial Crime Investigation Platform")

# ======================
# 📦 DATA SOURCE
# ======================
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Default Dataset", "Sample Large Data"]
)

def load_data():
    try:
        file = "fraud_data_large.csv" if dataset_option == "Sample Large Data" else "fraud_data.csv"
        data = pd.read_csv(file)
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

data = load_data()

if "text" not in data.columns:
    st.error("Dataset must contain 'text' column")
    st.stop()

# ======================
# 🔍 FEATURE EXTRACTION
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
# 🧠 RULE-BASED SCORE
# ======================
def rule_score(row):
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

data["rule_score"] = data.apply(rule_score, axis=1)

# ======================
# 🤖 ML PREP
# ======================
data["fraud_label"] = (
    (data["amount"] > 150000) &
    (data["txn_type"].isin(["TRANSFER", "CASH_OUT"]))
).astype(int)

data["txn_type_encoded"] = data["txn_type"].astype("category").cat.codes

X = data[["amount", "txn_type_encoded"]]
y = data["fraud_label"]

model = LogisticRegression()
model.fit(X, y)

# ======================
# 📊 MODEL EVALUATION
# ======================
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

# ======================
# ✅ ML SCORE
# ======================
data["risk_score"] = model.predict_proba(X)[:, 1] * 10

# ======================
# ✅ EXPLANATION FUNCTION
# ======================
def explain_prediction(amount, txn_type):
    explanation = []
    if amount > 100000:
        explanation.append("High transaction amount")
    if txn_type in ["TRANSFER", "CASH_OUT"]:
        explanation.append("Risky transaction type")
    if not explanation:
        explanation.append("No strong indicators")
    return explanation

# ======================
# 📥 SAVE CASE
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
# ✅ SAVE EVAL
# ======================
def save_evaluation(case_id, risk_score, decision):
    record = {
        "case_id": str(case_id),
        "risk_score": float(risk_score),
        "decision": decision,
        "timestamp": str(datetime.datetime.now())
    }
    with open("evaluation_log.json", "a") as f:
        f.write(json.dumps(record) + "\n")

# ======================
# ✅ FILTER
# ======================
suspicious_data = data[data["risk_score"] >= 7].sort_values(by="risk_score", ascending=False)

# ======================
# 📊 SIDEBAR
# ======================
st.sidebar.subheader("High Risk Cases")
st.sidebar.write(suspicious_data.head(10))

if len(suspicious_data) > 0:
    idx = st.sidebar.number_input("Select Case", 0, len(suspicious_data)-1, 0)

    if st.sidebar.button("Run Investigation"):
        row = suspicious_data.iloc[idx]

        alert = {
            "customer_id": f"C{idx}",
            "amount": row["amount"],
            "alert_type": row["txn_type"]
        }

        findings = []
        if row["amount"] > 100000:
            findings.append("High amount")
        if row["txn_type"] in ["TRANSFER", "CASH_OUT"]:
            findings.append("Risky type")

        report = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze fraud risk: {alert}, findings: {findings}"
        ).output_text

        st.session_state.case = {
            "text": row["text"],
            "alert": alert,
            "report": report,
            "risk_score": round(row["risk_score"], 2),
            "findings": findings,
            "case_id": f"C{idx}"
        }

# ======================
# 📄 CASE VIEW
# ======================
if "case" in st.session_state:

    case = st.session_state.case

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Case ID", case["case_id"])
    c2.metric("Amount", case["alert"]["amount"])
    c3.metric("Type", case["alert"]["alert_type"])
    c4.metric("Risk Score", case["risk_score"])

    st.subheader("📥 Transaction")
    st.write(case["text"])

    st.subheader("📄 Report")
    st.write(case["report"])

    st.subheader("🧠 Explanation")
    for e in explain_prediction(case["alert"]["amount"], case["alert"]["alert_type"]):
        st.write(f"• {e}")

    # QA
    q = st.text_input("Ask question")
    if st.button("Ask"):
        ans = client.responses.create(
            model="gpt-4.1-mini",
            input=f"{case} Question: {q}"
        )
        st.write(ans.output_text)

    # Decision
    decision = st.radio("Decision", ["Approve", "Reject", "Escalate"])

    if st.button("Submit Decision"):
        save_evaluation(case["case_id"], case["risk_score"], decision)
        st.success("Decision saved")

# ======================
# 📊 DASHBOARD
# ======================
st.subheader("📊 System Analytics")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    data["risk_score"].hist(ax=ax)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    data["txn_type"].value_counts().plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

# ======================
# 🤖 MODEL METRICS
# ======================
st.subheader("🤖 Model Performance")
st.write("Accuracy:", round(accuracy,2))
st.write("Precision:", round(precision,2))
st.write("Recall:", round(recall,2))

# ======================
# 🧠 RULE vs ML
# ======================
st.subheader("🧠 Rule vs ML Comparison")
st.write("Rule High:", len(data[data["rule_score"]>=7]))
st.write("ML High:", len(data[data["risk_score"]>=7]))

# ======================
# 🔍 DISAGREEMENT
# ======================
st.subheader("🔍 Disagreement Analysis")

disagree = data[(data["rule_score"]<7) & (data["risk_score"]>=7)]
st.write("ML caught but rules missed:", len(disagree))
st.dataframe(disagree.head())