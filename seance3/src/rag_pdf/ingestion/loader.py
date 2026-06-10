"""
Chargement de PDF avec conservation des numéros de page.

On utilise PyMuPDF (via PyMuPDFLoader) parce qu'il est rapide, fiable,
et qu'il extrait correctement les métadonnées de page — ce qui est
indispensable pour pouvoir afficher la page source dans l'app web.

Nouveauté séance 3 : sélection de pages — on peut ne charger qu'un
sous-ensemble du PDF (utile pour des gros documents où on veut indexer
qu'une partie ; piloté depuis l'app Streamlit).
"""
from pathlib import Path
from typing import Optional

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

    def load(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
    ) -> list[Document]:
        """Charge un PDF, optionnellement filtré sur certaines pages.

        Args:
            pdf_path: Chemin vers le PDF à charger.
            pages: Liste des numéros de page (1-indexés) à conserver.
                Si None, charge tout le document.

        Returns:
            Liste de Documents (un par page conservée) avec métadonnées normalisées.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si `pages` contient des numéros invalides.
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

        total_pages = len(documents)

        # Filtrage optionnel sur la sélection de pages.
        if pages is not None:
            invalid = [p for p in pages if p < 1 or p > total_pages]
            if invalid:
                raise ValueError(
                    f"Pages invalides : {invalid}. Le PDF a {total_pages} pages (1-{total_pages})."
                )
            pages_set = set(pages)
            documents = [d for d in documents if d.metadata["page"] in pages_set]
            logger.info(
                f"[green]✓[/] {len(documents)} pages chargées "
                f"(sélection : {len(pages_set)} pages sur {total_pages} au total)"
            )
        else:
            logger.info(f"[green]✓[/] {len(documents)} pages chargées (tout le document)")

        return documents
