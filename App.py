import streamlit as st
import requests
from groq import Groq
from Bio import Entrez

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

Entrez.email = "abommara@stevens.edu"

# ---------------- HELPER FUNCTIONS ----------------

def extract_drug_name(question):
    return question.lower().split()[-1]


def fetch_fda_data(query):
    try:
        url = f"https://api.fda.gov/drug/label.json?search={query}&limit=2"
        data = requests.get(url).json()

        texts = []
        for item in data.get("results", []):
            if "adverse_reactions" in item:
                texts.append(" ".join(item["adverse_reactions"]))

        return texts
    except:
        return []


def fetch_pubmed_data(query):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=2)
        record = Entrez.read(handle)
        ids = record["IdList"]

        texts = []

        if ids:
            fetch = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
            papers = Entrez.read(fetch)

            for article in papers["PubmedArticle"]:
                if "Abstract" in article["MedlineCitation"]["Article"]:
                    abstract_text = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                    texts.append(" ".join(abstract_text))

        return texts
    except:
        return []


def compute_confidence(sources):
    score = 0
    for s in sources:
        if s == "FDA":
            score += 0.5
        elif s == "PubMed":
            score += 0.3

    score = min(score, 1.0)

    if score > 0.75:
        level = "HIGH"
    elif score > 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, round(score, 2)


# ---------------- CORE FUNCTION ----------------

def ask_question(question):
    keyword = extract_drug_name(question)

    fda_data = fetch_fda_data(keyword)
    pubmed_data = fetch_pubmed_data(keyword)

    context = ""

    if fda_data:
        context += "FDA Data:\n" + "\n".join(fda_data) + "\n\n"

    if pubmed_data:
        context += "PubMed Data:\n" + "\n".join(pubmed_data)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""
Use ONLY the context below to answer.

Context:
{context}

Question:
{question}
"""
        }],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    sources = []
    if fda_data:
        sources.append("FDA")
    if pubmed_data:
        sources.append("PubMed")

    level, score = compute_confidence(sources)

    return answer, sources, level, score


# ---------------- UI ----------------

st.title("🧠 Medical Affairs AI Assistant")

question = st.text_input("Enter your medical question:")

if st.button("Ask"):
    if question:
        answer, sources, level, score = ask_question(question)

        st.subheader("🧾 Answer")
        st.write(answer)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Confidence Level", level)
        with col2:
            st.metric("Score", score)

        st.subheader("📚 Sources")
        for src in sources:
            st.write(f"- {src}")

    else:
        st.warning("Please enter a question.")
