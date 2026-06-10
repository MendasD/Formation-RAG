"""
Retriever — récupère les top-k chunks les plus proches d'une question.

Évolution séance 3 :
- Le retriever peut maintenant utiliser un reranker en post-traitement.
- Quand le reranker est actif, on remonte d'abord `top_k_initial` chunks
  (large), puis on les réordonne pour ne garder que `top_k` (précis).
- Optionnellement, on peut filtrer le retrieval sur certaines pages
  (utile pour l'app Streamlit).
"""
from typing import Optional

from langchain_core.documents import Document

from rag_pdf.indexing.vector_store import BaseVectorStore
from rag_pdf.retrieval.reranker import BaseReranker, NoOpReranker
from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """Encapsule la logique de récupération par similarité vectorielle + reranking optionnel.

    Args:
        vector_store: Le vector store à interroger.
        k: Nombre de chunks à retourner *in fine* (après reranking si activé).
        k_initial: Nombre de chunks remontés AVANT reranking. Si le reranker
            est `NoOpReranker`, ce paramètre est ignoré et on utilise `k`.
        reranker: Le reranker à appliquer en post-traitement (NoOp par défaut).
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        k: int = 4,
        k_initial: int = 20,
        reranker: Optional[BaseReranker] = None,
    ):
        self.vector_store = vector_store
        self.k = k
        self.k_initial = k_initial
        self.reranker = reranker or NoOpReranker()
        self._has_reranker = not isinstance(self.reranker, NoOpReranker)

    def retrieve(
        self,
        query: str,
        page_filter: Optional[list[int]] = None,
    ) -> list[Document]:
        """Renvoie les `k` chunks les plus pertinents pour la question.

        Args:
            query: La question (ou requête reformulée).
            page_filter: Si fourni, restreint la recherche à ces pages (1-indexées).
                Utile pour cibler une section spécifique du document.
        """
        # Étape 1 : retrieval initial. Si reranker actif, on remonte large.
        effective_k = self.k_initial if self._has_reranker else self.k
        docs = self.vector_store.similarity_search(query, k=effective_k)

        # Filtrage optionnel par pages.
        if page_filter is not None:
            page_set = set(page_filter)
            docs = [d for d in docs if d.metadata.get("page") in page_set]

        # Étape 2 : reranking (NoOp si désactivé → simple troncature à k).
        if self._has_reranker:
            docs = self.reranker.rerank(query, docs, top_k=self.k)
        else:
            docs = docs[: self.k]

        return docs

    def retrieve_with_scores(
        self, query: str
    ) -> list[tuple[Document, float]]:
        """Idem mais avec les scores de similarité bruts (sans reranking)."""
        return self.vector_store.similarity_search_with_score(query, k=self.k)
