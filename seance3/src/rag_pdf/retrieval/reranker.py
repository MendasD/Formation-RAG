"""
Reranking (post-retrieval).

Le retrieval initial (similarité vectorielle) est rapide mais imprécis :
deux chunks peuvent avoir des vecteurs proches sans pour autant répondre
à la question. Un reranker — un cross-encoder qui voit en même temps
la question ET le chunk — donne un score beaucoup plus précis, mais coûte
plus cher en calcul. On l'utilise donc en deux étapes :

1. Retrieval initial large : top_k_initial chunks (ex: 20)
2. Reranking précis : on garde le top_k final (ex: 4)

Modèle utilisé : BAAI/bge-reranker-v2-m3 — open-source, multilingue,
état de l'art sur MTEB.
"""
from abc import ABC, abstractmethod

from langchain_core.documents import Document

# Import différé de sentence_transformers.CrossEncoder (tire torch, ~5-10 s)
# → importé seulement quand on instancie effectivement CrossEncoderReranker.
from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Interface d'un reranker."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Réordonne les documents par pertinence et retourne le top_k.

        Args:
            query: La question (ou requête reformulée) servant de référence.
            documents: Les chunks remontés par le retrieval initial.
            top_k: Combien de chunks garder à la fin.

        Returns:
            Les top_k documents les plus pertinents selon le reranker.
            Chaque document conserve son metadata enrichi de `rerank_score`.
        """
        ...


class CrossEncoderReranker(BaseReranker):
    """Reranker basé sur un cross-encoder HuggingFace.

    Args:
        model_name: Nom du modèle (par défaut : `BAAI/bge-reranker-v2-m3`).
        device: 'cpu' ou 'cuda'.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        logger.info(f"[bold cyan]Chargement du reranker :[/] {model_name}")
        # Import paresseux : sentence_transformers tire torch.
        from sentence_transformers import CrossEncoder

        self._encoder = CrossEncoder(model_name, device=device)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        if not documents:
            return []

        # Cross-encoder : on lui passe des paires (query, chunk) et il sort un score.
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._encoder.predict(pairs)

        # On enrichit les métadonnées avec le score de reranking.
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top = scored[:top_k]
        for doc, score in top:
            doc.metadata["rerank_score"] = float(score)

        logger.info(
            f"[dim]Reranker :[/] {len(documents)} → {len(top)} chunks "
            f"(scores : {[round(float(s), 3) for _, s in top]})"
        )
        return [doc for doc, _ in top]


class NoOpReranker(BaseReranker):
    """Ne réordonne rien — équivalent à pas de reranker. Utilisé pour le baseline."""

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        return documents[:top_k]
