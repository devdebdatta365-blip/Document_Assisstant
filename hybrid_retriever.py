from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder, SentenceTransformer
from langchain_groq import ChatGroq
#Load once, reused across calls(don't reload on every question)
reranker_model=CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
sentence_model=SentenceTransformer("all-MiniLM-L6-v2")

def generate_hypothetical_answer(query):
    """
    Step 1 of HyDE: ask LLM to generate hypothetical answer to the query, without any document context

    """
    llm=ChatGroq(model="llama-3.1-8b-instant")
    response=llm.invoke(
        f"Write a short, factual answer (2-3 sentences) to this question. "
        f"Even if you are unsure, write a plausible answer:\n\n{query}"

    )
    return response.content
def get_hybrid_retriever(chunks, vector_store, k=20, rerank_top_n=5):
    """
    Takes your existing chunks + vector_store
    Returns a hybrid retriever (BM25 + dense, merged via RRF)
    """
    """
    Two-stage retrieval:
    1. Hybrid (BM25+Dense) retrieves top-k candidates
    2.Cross-encoder reranks those candidates, returns top rerank_top_n
    """
    # BM25 retriever - keyword based
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Dense retriever - embedding based (your existing ChromaDB)
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    # # Combine both with equal weight
    # hybrid_retriever = EnsembleRetriever(
    #     retrievers=[bm25_retriever, dense_retriever],
    #     weights=[0.5, 0.5]
    # )

    # return hybrid_retriever
    def hybrid_invoke(query):
        # Stage 1: HyDE
        hypothetical_answer=generate_hypothetical_answer(query)

        # Search vector store using hypothetical answer embedding
        hyde_docs=vector_store.similarity_search(
            hypothetical_answer,k=k
        )

        #Stage 2: BM25 on original query
        #BM25 still uses original query(keyword matching)
        bm25_docs=bm25_retriever.invoke(query)
        # dense_docs=dense_retriever.invoke(query) Used in hybrid ranking omitted when Hyde is used
        #RRF merge : BM25+ HyDE results
        scores={}
        for rank,doc in enumerate(bm25_docs):
            key=doc.page_content
            scores[key]=scores.get(key,0)+1/(rank+60)
        for rank,doc in enumerate(hyde_docs):
            key=doc.page_content
            scores[key]=scores.get(key,0)+1/(rank+60)
        all_docs={doc.page_content:doc for doc in bm25_docs+hyde_docs}
        candidates=sorted(
            all_docs.values(),
            key=lambda d:scores[d.page_content],
            reverse=True
            )[:k]
        
        #Reranking stage
        pairs=[[query,doc.page_content] for doc in candidates]
        rerank_scores=reranker_model.predict(pairs)

        reranked=sorted(
            zip(candidates,rerank_scores),
            key=lambda x:x[1],
            reverse=True
        )
        return [doc for doc, score in reranked[:rerank_top_n]],hypothetical_answer
    return hybrid_invoke