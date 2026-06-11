import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# ==============================
# 🔑 LOAD API KEY
# ==============================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: API key not found")
    exit()

# ✅ Fix SSL issue (corporate machine)
http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=api_key,
    http_client=http_client
)

print("\n🚀 AI Investigator Platform Started...\n")

# ==============================
# 🧾 ALERT INTAKE AGENT
# ==============================

alert = {
    "customer_id": "C102",
    "amount": 185000,
    "country": "Singapore",
    "alert_type": "High Velocity"
}

print("📥 --- ALERT RECEIVED ---")
print(alert)

# ==============================
# 👤 CUSTOMER INTELLIGENCE AGENT
# ==============================

def customer_intelligence_agent(customer_id):

    customer_db = {
        "C102": {
            "account_age": "2 years",
            "home_country": "India",
            "avg_transaction": 20000,
            "risk_segment": "Medium",
            "prior_alerts": 1
        }
    }

    return customer_db.get(customer_id, {})

customer_profile = customer_intelligence_agent(alert["customer_id"])

print("\n👤 --- CUSTOMER PROFILE ---")
print(customer_profile)

# ==============================
# ⚠️ PATTERN DETECTION AGENT
# ==============================

def pattern_detection_agent(alert):
    findings = []

    if alert["amount"] > 100000:
        findings.append("High transaction amount")

    if alert["country"] != "India":
        findings.append("Foreign transaction")

    if alert["alert_type"] == "High Velocity":
        findings.append("Rapid transaction activity")

    return findings

findings = pattern_detection_agent(alert)

print("\n⚠️ --- DETECTION FINDINGS ---")
print(findings)

# ==============================
# 📚 KNOWLEDGE RETRIEVAL AGENT
# ==============================

def knowledge_retrieval_agent(findings):

    knowledge_base = {
        "High transaction amount": "Transactions above 100k are classified as high risk and require enhanced due diligence.",
        "Foreign transaction": "Cross-border transactions are monitored under AML regulations due to higher laundering risk.",
        "Rapid transaction activity": "High velocity transactions may indicate structuring or fraud and should be escalated."
    }

    retrieved_knowledge = []

    for finding in findings:
        if finding in knowledge_base:
            retrieved_knowledge.append(knowledge_base[finding])

    return retrieved_knowledge

knowledge = knowledge_retrieval_agent(findings)

print("\n📘 --- KNOWLEDGE RETRIEVED ---")
for k in knowledge:
    print("-", k)

# ==============================
# 🧠 REASONING AGENT (AI)
# ==============================

def reasoning_agent(alert, findings, customer_profile, knowledge):

    prompt = f"""
You are a financial fraud investigator.

Alert:
{alert}

Customer Profile:
{customer_profile}

Findings:
{findings}

Relevant Policies:
{knowledge}

Generate a structured investigation report:

1. Risk Level
2. Key Reasons (behavioral + policy)
3. Recommendation
"""

    print("\n🤖 Calling AI Reasoning Agent...\n")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

report = reasoning_agent(alert, findings, customer_profile, knowledge)

print("\n📄 ===== INVESTIGATION REPORT =====\n")
print(report)

# ==============================
# 🔍 REFLECTION AGENT (SELF-CHECK)
# ==============================

def reflection_agent(report):

    prompt = f"""
You are a senior AML auditor.

Review this investigation report:

{report}

Check:

1. Logical consistency
2. Missing evidence
3. Weak reasoning
4. Policy alignment

Return:

- Validation Status (PASS / FAIL)
- Issues
- Suggested Improvements
"""

    print("\n🔍 Running Reflection Agent...\n")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

review = reflection_agent(report)

print("\n✅ ===== REFLECTION OUTPUT =====\n")
print(review)