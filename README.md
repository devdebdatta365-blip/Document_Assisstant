# 🧠 DocuMind AI - Document Assistant

A RAG-based (Retrieval-Augmented Generation) chatbot that answers 
questions about PDF documents intelligently, with conversational 
memory, source citations, and hybrid search.

> 🚧 This is an evolving project — built incrementally as part of 
> learning Enterprise RAG techniques. Check the Roadmap below.

---

## ✨ Features
- 📄 PDF upload and intelligent chunking
- 🔍 Hybrid Search (BM25 + Dense retrieval merged via Reciprocal Rank Fusion)
- 🧠 Conversational memory (multi-turn chat with history)
- 📚 Source attribution (shows which PDF pages the answer came from)
- ⚡ Fast inference via Groq (llama-3.1-8b-instant)
- 💾 Persistent vector store (ChromaDB) — embeddings computed once, reused on restart
- 🏠 Local embeddings via HuggingFace (all-MiniLM-L6-v2) — zero embedding API cost
- 📊 Retrieval comparison view (Dense vs Hybrid — see which pages each retrieves)

---

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace sentence-transformers |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| UI | Streamlit |

---

## ⚙️ Setup

1. Clone the repo
\```bash
git clone https://github.com/devdebdatta365-blip/Document_Assisstant.git
cd Document_Assisstant
\```

2. Install dependencies
\```bash
pip install -r requirements.txt
\```

3. Set up environment variables
\```bash
cp .env.example .env
# Add your GROQ_API_KEY inside .env
\```

4. Run the app
\```bash
streamlit run ui.py
\```

---

## 💡 How Hybrid Search Works

This project uses **Reciprocal Rank Fusion (RRF)** to combine 
BM25 and dense retrieval results:

- **BM25** — keyword-based, great for exact terms, model names, 
  error codes, specific numbers
- **Dense** — embedding-based, great for semantic meaning, 
  paraphrased questions, conceptual queries
- **RRF** — scores each document using `1/(rank + 60)` from both 
  lists and re-ranks the merged result

| Query Type | Example | Winner |
|---|---|---|
| Exact keyword | "What does HTTP 429 mean?" | BM25 / Hybrid |
| Semantic | "Why do AI models lie?" | Dense / Hybrid |
| Mixed | "Best DB for local dev?" | Hybrid |

---

## 📸 Screenshot
<img width="946" height="407" alt="image" src="https://github.com/user-attachments/assets/6a17ae84-89ef-45eb-ba7e-8147a37285a1" />


---

## 🗺️ Roadmap
- [x] Naive RAG pipeline (single document)
- [x] Hybrid Search (BM25 + Dense + RRF)
- [ ] Reranking (cross-encoder)
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] CRAG (Corrective RAG with relevance grading)
- [ ] Self-RAG (self-critique loop)
- [ ] Text2SQL module
- [ ] Semantic caching
- [ ] Guardrails + RAGAS evaluation


