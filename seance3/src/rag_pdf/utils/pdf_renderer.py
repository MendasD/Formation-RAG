"""
Rendu d'une page PDF en image — pour l'affichage dans l'app Streamlit.

PyMuPDF (fitz) permet d'extraire chaque page d'un PDF comme une image PNG
de qualité réglable. C'est ce qui nous permet d'afficher visuellement la
page source d'un chunk dans le panneau de droite de l'app.
"""
from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF


@lru_cache(maxsize=128)
def render_page_to_png(pdf_path: str, page_number: int, dpi: int = 150) -> bytes:
    """Rend une page PDF en PNG, retourné comme bytes.

    Args:
        pdf_path: Chemin (string pour être hashable par lru_cache) vers le PDF.
        page_number: Numéro de page **1-indexé** (comme dans le reste du projet).
        dpi: Résolution. 150 = bon compromis qualité/taille. 200+ = très net.

    Returns:
        Les bytes du PNG, prêts à être passés à `st.image()` de Streamlit.

    Raises:
        FileNotFoundError: Si le PDF n'existe pas.
        ValueError: Si le numéro de page est hors limites.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    doc = fitz.open(p)
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(
                f"Page {page_number} hors limites (PDF de {doc.page_count} pages)"
            )

        # PyMuPDF utilise un index 0-based en interne.
        page = doc.load_page(page_number - 1)

        # On rend à la résolution demandée. zoom = dpi / 72 (PDF natif = 72 DPI).
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
    finally:
        doc.close()


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Retourne le nombre total de pages d'un PDF."""
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()
