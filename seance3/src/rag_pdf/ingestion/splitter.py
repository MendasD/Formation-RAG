"""
Découpage des documents en chunks.

Stratégie utilisée : **RecursiveCharacterTextSplitter**.
C'est la stratégie par défaut recommandée pour 80% des cas :
elle découpe en respectant la hiérarchie naturelle du texte
(paragraphes > phrases > mots > caractères).
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class Splitter:
    """Découpe une liste de Documents en chunks de taille contrôlée.

    Args:
        chunk_size: Taille cible d'un chunk (en caractères).
        chunk_overlap: Chevauchement entre chunks consécutifs (en caractères).
            Important pour ne pas perdre l'info qui se trouve à la frontière
            entre deux chunks.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Ordre essayé : double saut de ligne, simple saut, fin de phrase, espace, caractère.
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Découpe les documents en chunks en préservant leurs métadonnées."""
        chunks = self._splitter.split_documents(documents)
        logger.info(
            f"[green]✓[/] {len(chunks)} chunks créés "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
