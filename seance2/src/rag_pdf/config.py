"""
Configuration centralisée du projet RAG.

Toute la configuration vit ici. Les paramètres sont chargés depuis :
1. Les variables d'environnement (priorité maximale)
2. Le fichier .env à la racine de seance2/
3. Les valeurs par défaut définies dans cette classe

Cela garantit qu'aucun paramètre métier n'est codé en dur dans le reste
du package — un principe fondamental d'une architecture propre.

NB : on calcule un chemin **absolu** vers le fichier .env (et non un chemin
relatif au CWD), pour que la config se charge correctement quelle que soit
la façon dont on lance le code (notebook depuis notebooks/, script depuis
seance2/, ou import depuis n'importe où ailleurs).
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet seance2/ — calculée depuis l'emplacement de ce fichier :
# src/rag_pdf/config.py  →  .parent.parent.parent  =  seance2/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Configuration du système RAG, validée au démarrage."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Secrets / clés API
    # ------------------------------------------------------------------
    groq_api_key: str = Field(
        ...,
        description="Clé API Groq, obtenue sur https://console.groq.com/keys",
    )

    # ------------------------------------------------------------------
    # Chemins
    # ------------------------------------------------------------------
    project_root: Path = Field(default=_PROJECT_ROOT)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "pdfs"

    @property
    def chroma_persist_dir(self) -> Path:
        return self.project_root / "chroma_db"

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        description="Modèle HuggingFace d'embeddings (multilingue, support FR).",
    )
    embedding_device: str = Field(
        default="cpu",
        description="'cpu' ou 'cuda' si GPU disponible.",
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=800, description="Taille cible d'un chunk (caractères).")
    chunk_overlap: int = Field(default=100, description="Chevauchement entre chunks consécutifs.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(default=7, description="Nombre de chunks récupérés par requête.")
    collection_name: str = Field(default="rag_pdf_notebook", description="Nom de la collection Chroma.")

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Modèle Groq à utiliser pour la génération.",
    )
    llm_temperature: float = Field(default=0.0, description="0 = déterministe, 1 = créatif.")


# Instance unique réutilisée dans tout le projet.
settings = Settings()
