import streamlit as st
import os
import requests
import spacy

from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from Bio import Entrez

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

# Load API Key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# PubMed email (required)
Entrez.email = "your_email@example.com"

# ---------------- CORE FUNCTIONS ----------------

def extract_drug_name(question):
    doc = nlp(question)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            return ent.text.lower()
    return question.split()[-1].lower()


def fetch_fda_data(query):
    try:
        url = f"https://api.fda.gov/drug/label.json?search={query}&limit=2"
        data = requests.get(url).json()

        docs = []
        for item in data.get("results", []):
            text = ""
            if "adverse_reactions" in item:
                text += "Adverse Reactions:\n" + " ".join(item["adverse_reactions"])

            if text:
                docs.append(Document(page_content=text, metadata={"source": "FDA"}))

        return docs
    except:
        return []


def fetch_pubmed_data(query):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=2)
        record = Entrez.read(handle)
        ids = record["IdList"]

        docs = []

        if ids:
            fetch = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
            papers = Entrez.read(fetch)

            for article in papers["PubmedArticle"]:
                abstract = ""

                if "Abstract" in article["MedlineCitation"]["Article"]:
                    abstract_text = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                    abstract = " ".join(abstract_text)

                if abstract:
                    docs.append(
                        Document(
                            page_content=abstract,
                            metadata={"source": "PubMed"}
                        )
                    )

        return docs
    except:
        return []


def compute_confidence(docs):
    sources = [doc.metadata.get("source", "") for doc in docs]

    score = 0
    for s in sources:
        if "FDA" in s:
            score += 0.5
        elif "Internal" in s:
            score += 0.3
        else:
            score += 0.2

    score = min(score, 1.0)

    agreement = "CONSISTENT" if len(set(sources)) > 1 else "LIMITED"

    if score > 0.75:
        level = "HIGH"
    elif score > 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "score": round(score, 2),
        "level": level,
        "agreement": agreement,
        "sources": list(set(sources))
    }


def get_context_with_sources(docs):
    return "\n\n".join([
        f"[Source {i+1}: {d.metadata['source']}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ])


# ---------------- RAG SETUP ----------------

internal_text = """
Remdesivir adverse effects include nausea and elevated liver enzymes.
Contraindicated in severe liver dysfunction.
"""

documents = [
    Document(page_content=internal_text, metadata={"source": "Internal"})
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embedding_model)


def ask_question(question):
    docs = vectorstore.similarity_search(question, k=2)

    keyword = extract_drug_name(question)

    fda_docs = fetch_fda_data(keyword)
    pubmed_docs = fetch_pubmed_data(keyword)

    all_docs = docs + fda_docs + pubmed_docs

    context = get_context_with_sources(all_docs)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": context + "\n\nQuestion: " + question}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    confidence = compute_confidence(all_docs)

    return {"answer": answer, "confidence": confidence}


# ---------------- UI ----------------

st.title("🧠 Medical Affairs AI Assistant")

question = st.text_input("Enter your medical question:")

if st.button("Ask"):
    if question:
        result = ask_question(question)

        st.subheader("🧾 Answer")
        st.write(result["answer"])

        conf = result["confidence"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Confidence Level", conf["level"])
        with col2:
            st.metric("Score", conf["score"])
        with col3:
            st.metric("Agreement", conf["agreement"])

        st.subheader("📚 Sources")
        for src in conf["sources"]:
            st.write(f"- {src}")
    else:
        st.warning("Please enter a question.")
