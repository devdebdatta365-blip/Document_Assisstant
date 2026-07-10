# 🧠 DocuMind AI - Document Assistant

A RAG-based (Retrieval-Augmented Generation) chatbot that answers 
questions about PDF documents intelligently, with conversational 
memory, source citations, and a three-stage retrieval pipeline.

> 🚧 This is an evolving project — built incrementally as part of 
> learning Enterprise RAG techniques. Check the Roadmap below.

---

## ✨ Features
- 📄 PDF upload and intelligent chunking
- 🔍 Hybrid Search (BM25 + Dense retrieval merged via Reciprocal Rank Fusion)
- 💡 HyDE (Hypothetical Document Embeddings) — query enhancement for better recall
- 🎯 Cross-encoder reranking (ms-marco-TinyBERT-L-2-v2) — higher precision
- 🧠 Conversational memory (multi-turn chat with history)
- 📚 Source attribution (shows which PDF pages the answer came from)
- ⚡ Fast inference via Groq (llama-3.1-8b-instant)
- 💾 Persistent vector store (ChromaDB) — embeddings computed once, reused on restart
- 🏠 Local embeddings via HuggingFace (all-MiniLM-L6-v2) — zero embedding API cost
- 📊 Retrieval comparison view (Dense vs Final Reranked+HyDE)

---

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | sentence-transformers (ms-marco-TinyBERT-L-2-v2) |
| UI | Streamlit |

---

## ⚙️ Setup

1. Clone the repo
```bash
git clone https://github.com/devdebdatta365-blip/Document_Assisstant.git
cd Document_Assisstant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Add your GROQ_API_KEY inside .env
```

4. Run the app
```bash
streamlit run ui.py
```

---

## 🔬 How the Three-Stage Retrieval Pipeline Works

### Stage 1 — HyDE (Query Enhancement)
Instead of embedding the raw query, the LLM first generates a 
hypothetical answer. That answer uses vocabulary closer to the 
document, so the embedding lands nearer to the real answer chunk 
in vector space.

### Stage 2 — Hybrid Search (BM25 + Dense + RRF)
Both BM25 (keyword) and dense (HyDE-enhanced embedding) search 
run in parallel. Results are merged using Reciprocal Rank Fusion:
`score = 1 / (rank + 60)` — chunks appearing high in both lists 
get the highest combined score.

- **BM25** — exact terms, model names, error codes, numbers
- **Dense + HyDE** — semantic meaning, paraphrased questions, concepts
- **RRF** — best of both, 20 candidates passed to reranker

### Stage 3 — Cross-Encoder Reranking
The cross-encoder scores each candidate by reading the query and 
chunk *together* (not as separate embeddings), giving a much more 
accurate relevance score. Top 5 results are sent to the LLM.

```
Query
  │
  ├── HyDE: LLM generates hypothetical answer
  │         └── embed → dense search (k=20)
  │
  ├── BM25: keyword search on original query (k=20)
  │
  ├── RRF merge → 20 candidates
  │
  └── Cross-encoder reranker → Top 5 → LLM
```

**Why two stages of retrieval + reranking?**
- Stage 1+2 (Hybrid+HyDE): optimises for *recall* — don't miss anything relevant
- Stage 3 (Reranking): optimises for *precision* — put the best ones first

---

## 📊 Retrieval Comparison Results

| Question | Dense pages | Final Reranked+HyDE pages | What helped |
|---|---|---|---|
| "Why ChromaDB for local, Qdrant for production?" | [1] | [0, 1, 2] | Reranking caught 2 missed pages |
| "What happens when a model doesn't know?" | [1] | [0, 1] | HyDE caught vocabulary mismatch |
| "What does HTTP 429 mean?" | [0] | [0] | Dense already correct (exact keyword) |

---

## 📸 Screenshots

**Reranking in action:**

<img width="946" height="407" alt="image" src="https://github.com/user-attachments/assets/6a17ae84-89ef-45eb-ba7e-8147a37285a1" />

> Dense retrieval found only page [1]. Reranking found all 3 relevant 
> pages — enabling a complete answer about ChromaDB vs Qdrant.

---

**HyDE in action:**

<img width="1568" height="670" alt="image" src="https://github.com/user-attachments/assets/43ed7719-1c6f-4bfb-a509-25a7b81996f5" />

> Query *"What happens when a language model doesn't know the answer?"* 
> contains no exact vocabulary from the document. Dense retrieval found 
> only page [1]. HyDE generated a hypothetical answer using relevant 
> terms ("hallucination", "training data"), retrieved page [0] that 
> dense missed — enabling a more complete final answer.

---

## 🗺️ Roadmap
- [x] Naive RAG pipeline (single document)
- [x] Hybrid Search (BM25 + Dense + RRF)
- [x] HyDE (Hypothetical Document Embeddings)
- [x] Cross-encoder Reranking (two-stage retrieval)
- [ ] CRAG (Corrective RAG with relevance grading)
- [ ] Self-RAG (self-critique loop)
- [ ] Text2SQL module
- [ ] Semantic caching
- [ ] Guardrails + RAGAS evaluation

---
