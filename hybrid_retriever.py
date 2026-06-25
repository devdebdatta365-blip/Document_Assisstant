from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_hybrid_retriever(chunks, vector_store, k=5):
    """
    Takes your existing chunks + vector_store
    Returns a hybrid retriever (BM25 + dense, merged via RRF)
    """
    # BM25 retriever - keyword based
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Dense retriever - embedding based (your existing ChromaDB)
    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    # Combine both with equal weight
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever