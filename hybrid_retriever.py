from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder 

#Load once, reused across calls(don't reload on every question)
reranker_model=CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
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
        bm25_docs=bm25_retriever.invoke(query)
        dense_docs=dense_retriever.invoke(query)
        #RRF merge
        scores={}
        for rank,doc in enumerate(bm25_docs):
            key=doc.page_content
            scores[key]=scores.get(key,0)+1/(rank+60)
        for rank,doc in enumerate(dense_docs):
            key=doc.page_content
            scores[key]=scores.get(key,0)+1/(rank+60)
        all_docs={doc.page_content:doc for doc in bm25_docs+dense_docs}
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
        return [doc for doc, score in reranked[:rerank_top_n]]
    return hybrid_invoke