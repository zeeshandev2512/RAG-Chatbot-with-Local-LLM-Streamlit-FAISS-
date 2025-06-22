import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import requests

# --- Load vector DB and chunks ---
index = faiss.read_index("faiss_index.index")
with open("document_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# --- Load embedding model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_query(query):
    return embed_model.encode([query])[0].astype("float32")

def retrieve_chunks(query, top_k=3):
    vector = embed_query(query).reshape(1, -1)
    _, indices = index.search(vector, top_k)
    return [chunks[i] for i in indices[0]]

def build_prompt(retrieved_chunks, user_query):
    context = "\n\n".join(retrieved_chunks)
    return f"""You are an AI assistant answering questions based on the provided document context.

Context:
{context}

Question:
{user_query}

Answer:"""

def query_ollama(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def rag_chatbot(query, top_k=3):
    retrieved = retrieve_chunks(query, top_k)
    prompt = build_prompt(retrieved, query)
    response = query_ollama(prompt)
    return response, retrieved

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 AI Chatbot | RAG + Ollama")

query = st.text_input("Ask a question about the document:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        response, sources = rag_chatbot(query)
    st.markdown("### 🧠 Response:")
    st.write(response)

    with st.expander("📄 Source Chunks Used"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")

with st.sidebar:
    st.markdown("### ⚙️ App Info")
    st.markdown("**Model:** mistral via Ollama")
    st.markdown(f"**Chunks in DB:** {len(chunks)}")
    if st.button("🔄 Reset Chat"):
        st.experimental_rerun()
