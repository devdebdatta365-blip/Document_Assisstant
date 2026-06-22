# Document Assistant

A RAG-based chatbot that answers questions about a PDF document, with 
conversational memory and source citations (page numbers).

Built as the first milestone toward a larger Enterprise RAG project 
(hybrid search, reranking, CRAG/Self-RAG, multi-document support — 
in progress).

## Features
- PDF ingestion and chunking (RecursiveCharacterTextSplitter)
- Persistent vector store (ChromaDB) — embeddings computed once, reused on restart
- Local embeddings via HuggingFace (`all-MiniLM-L6-v2`) — no embedding API cost
- Fast inference via Groq (`llama-3.1-8b-instant`)
- Conversational memory (multi-turn chat with history)
- Source attribution (shows which PDF pages the answer came from)

## Tech Stack
- LangChain
- ChromaDB (vector store)
- Groq API (LLM)
- HuggingFace sentence-transformers (embeddings)
- Streamlit (UI) / CLI (app.py)

## Setup
\`\`\`bash
git clone https://github.com/devdebdatta365-blip/Document_Assisstant.git
cd Document_Assisstant
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
\`\`\`

## Usage
CLI version:
\`\`\`bash
python app.py
\`\`\`
Streamlit UI:
\`\`\`bash
streamlit run ui.py
\`\`\`
Replace `sample.pdf` with your own document, or extend to file upload (see Roadmap).

## Example
**Q:** [paste a real question you tested]
**A:** [paste the real answer]
Sources: Page X, Page Y

## Roadmap
- [ ] Multi-document support
- [ ] File upload instead of hardcoded `sample.pdf`
- [ ] Hybrid search (BM25 + dense retrieval)
- [ ] Reranking
- [ ] CRAG / Self-RAG
- [ ] Semantic caching
- [ ] Evaluation with RAGAS
