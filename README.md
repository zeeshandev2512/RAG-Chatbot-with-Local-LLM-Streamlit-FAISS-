# RAG Chatbot with Local LLM (Streamlit + FAISS)

This project is an AI-powered chatbot that can answer questions based on a custom document using a Retrieval-Augmented Generation (RAG) pipeline. It combines local embeddings, semantic search, and a local LLM via Ollama, all wrapped in a simple Streamlit interface.

---

## 📄 Project Overview

* **Goal**: Answer user questions based on a provided document (e.g., terms, policies, legal contracts).
* **Tech Stack**: Python, Streamlit, SentenceTransformers, FAISS, Ollama (Mistral), spaCy
* **Key Features**:

  * Document chunking and preprocessing
  * FAISS vector search
  * Prompt formatting for context-aware generation
  * Streamlit interface with real-time response streaming

---

## 📁 Directory Structure

```
├── app.py                     # Streamlit frontend
├── document_chunks.pkl       # Saved text chunks
├── faiss_index.index         # Vector store index
├── requirements.txt          # Python dependencies
├── README.md
└── AI Training Document.pdf  # Input document
```

---

## 🚀 Demo

### 🔗 Watch the Demo Video:

[Demo Video Link Here](https://your-demo-link.com)

### 🔹 Screenshot or GIF

![Chatbot Demo](https://your-gif-or-demo-link.gif)

---

## 🛠️ How It Works

1. **Text Extraction**: PDF is cleaned and split into sentence-based chunks (100–300 words).
2. **Embeddings**: Each chunk is embedded using `all-MiniLM-L6-v2`.
3. **Vector Indexing**: FAISS stores all chunk embeddings.
4. **Query Flow**:

   * User enters a question
   * Query is embedded and matched to top-k chunks
   * These are injected into a prompt template
   * Prompt is passed to Ollama (Mistral model) for generation

---

## 📥 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Local LLM

```bash
ollama run mistral
```

### 4. Launch the App

```bash
python -m streamlit run app.py
```

---

## 💡 Example Queries

* What are the objectives of the AI training document?
* What models can be used?
* How is the evaluation done?
* What is the meaning of life? *(hallucination example)*

---

## ⚙️ Tech Details

* **Embedding Model**: `all-MiniLM-L6-v2`
* **LLM**: Mistral 7B via Ollama
* **Vector DB**: FAISS (L2 distance)
* **Frontend**: Streamlit

---

## ⚠️ Known Limitations

* No confidence scoring or citation highlighting
* Can hallucinate for off-topic queries
* Performance limited by CPU (slower than GPU LLMs)

---

## 📌 To-Do / Future Improvements

* Add source highlighting or reference tags
* Support longer documents with chunk overlap
* Add streaming token-by-token response
* Dockerize the setup for easier deployment

---

## 🙋 Contact

For questions or feedback, feel free to reach out via [GitHub Issues](https://github.com/yourusername/rag-chatbot/issues).

---

## 📄 License

This project is open-source and free to use under the MIT License.
