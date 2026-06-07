"""
Vector store — Interface + implémentation Chroma persistante.

ChromaDB est parfait pour démarrer : open-source, simple, persistance
sur disque locale, zéro infrastructure. En séance 3 on montrera comment
swapper vers Pinecone en quelques lignes — d'où l'intérêt de l'interface.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_pdf.indexing.embedder import BaseEmbedder
from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """Interface d'un vector store."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        ...

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @property
    @abstractmethod
    def langchain_store(self) -> VectorStore:
        ...


class ChromaVectorStore(BaseVectorStore):
    """Vector store ChromaDB persistant sur disque.

    Args:
        embedder: Le fournisseur d'embeddings à utiliser (même au moment
            de l'indexation et de la requête — c'est impératif !).
        persist_directory: Dossier où Chroma sauvegardera la base.
        collection_name: Nom de la collection (équivalent d'une "table").
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        persist_directory: str | Path,
        collection_name: str = "rag_pdf",
    ):
        self._persist_dir = Path(persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder.langchain_embeddings,
            persist_directory=str(self._persist_dir),
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Indexe une liste de documents (idempotent côté chroma_db sur le contenu)."""
        self._store.add_documents(documents)
        logger.info(
            f"[green]✓[/] {len(documents)} chunks indexés "
            f"(total dans la collection : {self.count()})"
        )

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Comme `similarity_search` mais retourne aussi le score (distance).

        Note: dans Chroma, plus le score est *bas*, plus le chunk est proche
        (c'est une distance, pas une similarité).
        """
        return self._store.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        """Nombre total de chunks dans la collection."""
        return self._store._collection.count()

    @property
    def langchain_store(self) -> VectorStore:
        return self._store
