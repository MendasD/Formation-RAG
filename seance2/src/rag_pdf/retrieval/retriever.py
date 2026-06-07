"""
Retriever — récupère les top-k chunks les plus proches d'une question.

C'est volontairement une classe très simple en séance 2 (Naive RAG).
En séance 3 on ajoutera à ce dossier :
    - query_rewriter.py : reformulation de la question (pré-retrieval)
    - reranker.py       : ré-ordonnancement précis (post-retrieval)
"""
from langchain_core.documents import Document

from rag_pdf.indexing.vector_store import BaseVectorStore


class Retriever:
    """Encapsule la logique de récupération par similarité vectorielle."""

    def __init__(self, vector_store: BaseVectorStore, k: int = 4):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str) -> list[Document]:
        """Renvoie les `k` chunks les plus proches de la question."""
        return self.vector_store.similarity_search(query, k=self.k)

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Idem mais avec les scores de similarité (utile pour debug/UI)."""
        return self.vector_store.similarity_search_with_score(query, k=self.k)
