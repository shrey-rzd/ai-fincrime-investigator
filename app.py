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
# ✅ SESSION STATE
# ======================
if "case_data" not in st.session_state:
    st.session_state.case_data = None

# ======================
# 🔐 API KEY
# ======================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))

st.title("🚨 Financial Crime Investigation Platform")

# ======================
# ✅ DATA SOURCE
# ======================
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Default Dataset", "Sample Large Data"]
)

def load_data():
    try:
        file = "fraud_data_large.csv" if dataset_option == "Sample Large Data" else "fraud_data.csv"
        df = pd.read_csv(file)
        return df.dropna()
    except:
        return pd.DataFrame()

data = load_data()

if "text" not in data.columns:
    st.error("Dataset must contain 'text'")
    st.stop()

# ======================
# ✅ RAG: LOAD DOCS
# ======================
def load_documents():
    docs = []

    try:
        with open("fraud_policy.txt", "r") as f:
            docs.append(f.read())
    except:
        pass

    try:
        with open("case_history_structured.json", "r") as f:
            cases = json.load(f)
            for c in cases:
                docs.append(str(c))
    except:
        pass

    return docs

documents = load_documents()

# ======================
# ✅ CHUNKING (IMPROVED)
# ======================
def chunk_text(text):
    chunks = text.split("\n")
    return [c.strip() for c in chunks if c.strip()]

knowledge_base = []
for doc in documents:
    knowledge_base.extend(chunk_text(doc))

# ======================
# ✅ RETRIEVER
# ======================
def retrieve_chunks(query, top_k=3):
    results = []
    query_words = set(query.lower().split())

    for chunk in knowledge_base:
        score = len(query_words.intersection(set(chunk.lower().split())))
        results.append((score, chunk))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return [c for s, c in results[:top_k] if s > 0]

# ======================
# ✅ FEATURE EXTRACTION
# ======================
def extract(text):
    amt = float(re.search(r"\d+", text).group()) if re.search(r"\d+", text) else 0
    if "PAYMENT" in text:
        t = "PAYMENT"
    elif "TRANSFER" in text:
        t = "TRANSFER"
    elif "CASH_OUT" in text:
        t = "CASH_OUT"
    else:
        t = "OTHER"
    return amt, t

data["amount"] = data["text"].apply(lambda x: extract(x)[0])
data["txn_type"] = data["text"].apply(lambda x: extract(x)[1])

# ======================
# ✅ RULE SCORE
# ======================
def rule_score(r):
    s = 0
    if r.amount > 200000: s += 6
    elif r.amount > 100000: s += 4
    elif r.amount > 50000: s += 2

    if r.txn_type == "CASH_OUT": s += 4
    elif r.txn_type == "TRANSFER": s += 3
    elif r.txn_type == "PAYMENT": s += 1

    return min(s, 10)

data["rule_score"] = data.apply(rule_score, axis=1)

# ======================
# ✅ ML MODEL
# ======================
data["fraud_label"] = ((data.amount > 150000) & data.txn_type.isin(["TRANSFER","CASH_OUT"])).astype(int)
data["txn_type_encoded"] = data.txn_type.astype("category").cat.codes

X = data[["amount","txn_type_encoded"]]
y = data["fraud_label"]

model = LogisticRegression()
model.fit(X,y)

y_pred = model.predict(X)

accuracy = accuracy_score(y,y_pred)
precision = precision_score(y,y_pred)
recall = recall_score(y,y_pred)

data["risk_score"] = model.predict_proba(X)[:,1]*10

# ======================
# ✅ FILTER
# ======================
suspicious = data[data.risk_score >=7].sort_values("risk_score",ascending=False)

st.sidebar.subheader("High Risk Cases")
st.sidebar.write(suspicious.head(10))

if len(suspicious)>0:
    idx = st.sidebar.number_input("Select Case",0,len(suspicious)-1,0)

    if st.sidebar.button("Run Investigation"):
        row = suspicious.iloc[idx]

        retrieved = retrieve_chunks(row.text)
        context = "\n\n".join(retrieved)

        prompt = f"""
Transaction:
{row.text}

Context:
{context}

Give risk, reasons, recommendation.
"""

        report = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        ).output_text

        st.session_state.case_data = {
            "text":row.text,
            "amount":row.amount,
            "type":row.txn_type,
            "risk":round(row.risk_score,2),
            "report":report
        }

# ======================
# ✅ CASE VIEW
# ======================
if st.session_state.case_data:

    c = st.session_state.case_data

    a,b,c3,c4 = st.columns(4)
    a.metric("Amount",c["amount"])
    b.metric("Type",c["type"])
    c3.metric("Risk",c["risk"])

    st.subheader("📄 Report")
    st.write(c["report"])

    st.subheader("🧠 Explanation")
    if c["amount"]>100000:
        st.write("• High amount")
    if c["type"] in ["TRANSFER","CASH_OUT"]:
        st.write("• Risky type")

    # ✅ QA AGENT
    st.subheader("💬 Ask Investigator")
    q = st.text_input("Ask Question")

    if st.button("Ask") and q:
        ctx = "\n\n".join(retrieve_chunks(q))

        ans = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Question:{q}\nContext:{ctx}"
        )
        st.info(ans.output_text)

    # ✅ DECISION
    st.subheader("👨‍💼 Decision")
    decision = st.radio("",["Approve","Reject","Escalate"])

    if st.button("Submit Decision"):
        with open("decisions_log.txt","a") as f:
            f.write(f"{datetime.datetime.now()} - {decision}\n")
        st.success("Saved ✅")

# ======================
# 📊 DASHBOARD
# ======================
st.subheader("📊 Analytics")

col1,col2 = st.columns(2)

with col1:
    fig,ax = plt.subplots()
    data.risk_score.hist(ax=ax)
    st.pyplot(fig)

with col2:
    fig,ax = plt.subplots()
    data.txn_type.value_counts().plot(kind="bar",ax=ax)
    st.pyplot(fig)

# ======================
# 🤖 METRICS
# ======================
st.subheader("Model Performance")
st.write("Accuracy:",round(accuracy,2))
st.write("Precision:",round(precision,2))
st.write("Recall:",round(recall,2))

# ======================
# 🧠 RULE VS ML
# ======================
st.subheader("Rule vs ML")
st.write("Rule High:",len(data[data.rule_score>=7]))
st.write("ML High:",len(data[data.risk_score>=7]))

# ======================
# 🔍 DISAGREEMENT
# ======================
st.subheader("Disagreement")
d = data[(data.rule_score<7)&(data.risk_score>=7)]
st.write("ML caught:",len(d))
st.dataframe(d.head())

# ======================
# 🛠 DEBUG
# ======================
with st.expander("Debug RAG"):
    st.write("Docs:",len(documents))
    st.write("Chunks:",len(knowledge_base))
