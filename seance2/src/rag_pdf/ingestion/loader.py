"""
Chargement de PDF avec conservation des numéros de page.

On utilise PyMuPDF (via PyMuPDFLoader) parce qu'il est rapide, fiable,
et qu'il extrait correctement les métadonnées de page — ce qui est
indispensable pour pouvoir afficher la page source dans l'app web.
"""
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """Charge un PDF en une liste de `Document` LangChain (un par page).

    Chaque document porte dans `metadata` :
        - source : le nom du fichier PDF
        - page : le numéro de page (1-indexé, plus naturel pour l'utilisateur)
    """

    def load(self, pdf_path: str | Path) -> list[Document]:
        """Charge un PDF.

        Args:
            pdf_path: Chemin (str ou Path) vers le PDF à charger.

        Returns:
            Liste de Documents (un par page) avec métadonnées normalisées.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        logger.info(f"[bold cyan]Chargement du PDF :[/] {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()

        # Normalisation des métadonnées : on garde uniquement ce qui nous sert,
        # et on convertit la page en 1-indexé (PyMuPDF utilise 0-indexé).
        for doc in documents:
            original_page = doc.metadata.get("page", 0)
            doc.metadata = {
                "source": pdf_path.name,
                "page": original_page + 1,
            }

        logger.info(f"[green]✓[/] {len(documents)} pages chargées")
        return documents
