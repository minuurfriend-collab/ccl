import streamlit as st
import json
import faiss
import numpy as np
import requests
import os
import random
from sample_questions import SAMPLE_QUESTIONS
from sentence_transformers import SentenceTransformer

def call_llm(prompt):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful Indian legal assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"API Error: {response.status_code} - {response.text}"

    result = response.json()
    return result["choices"][0]["message"]["content"]

# -----------------------------
# Load resources (cached)
# -----------------------------
@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open("corpus.json", "r") as f:
        corpus = json.load(f)

    # Convert structured entries into searchable text
    documents = [
        f"{item['act']} Section {item['section']} - {item['title']}. {item['text']}"
        for item in corpus
    ]

    # Create embeddings
    embeddings = model.encode(documents, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(np.array(embeddings, dtype=np.float32))

    return model, index, corpus, documents


MODEL, INDEX, CORPUS, DOCUMENTS = load_resources()

if "query" not in st.session_state:
    st.session_state.query = ""

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=3):
    q_emb = MODEL.encode([query], normalize_embeddings=True)
    scores, idxs = INDEX.search(np.array(q_emb, dtype=np.float32), top_k)
    return [CORPUS[i] for i in idxs[0]]


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Legal RAG Assistant", layout="centered")

st.title("⚖️ Indian Legal Assistant – RAG Powered")
st.caption("IPC • RTI Act • Consumer Protection Act")

st.markdown("""
This system uses **Retrieval-Augmented Generation (RAG)**  
to provide grounded, citation-based legal information.

⚠️ *Academic demo only. Not legal advice.*
""")

if "chat" not in st.session_state:
    st.session_state.chat = []

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Pre-generated question", on_click=pick_random_question):
        st.session_state.query = random.choice(SAMPLE_QUESTIONS)

with col2:
    st.caption("Click to auto-fill a sample legal question for demo")

query = st.text_input("Ask a legal question", key="query")


top_k = st.slider("Top-K Sections", 1, 5, 3)

if st.button("Submit") and query:
    results = retrieve(query, top_k)

    context = "\n\n".join([
        f"{r['act']} Section {r['section']} – {r['title']}\n{r['text']}"
        for r in results
    ])

if st.button("🎲 Regenerate & Run"):
    st.session_state.query = random.choice(SAMPLE_QUESTIONS)
    st.session_state.auto_submit = True

    prompt = f"""
You are an expert Indian legal assistant.

Answer strictly using ONLY the legal context provided below.
Cite act names and section numbers clearly.
Do not fabricate laws.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = call_llm(prompt)

    st.session_state.chat.append((query, answer))


# Display conversation
for q, a in reversed(st.session_state.chat):
    st.markdown(f"**You:** {q}")
    st.markdown(a)

st.write("Token loaded:", os.environ.get("HF_TOKEN") is not None)
