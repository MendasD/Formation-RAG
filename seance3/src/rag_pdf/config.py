"""
Configuration centralisée du projet RAG.

Toute la configuration vit ici. Les paramètres sont chargés depuis :
1. Les variables d'environnement (priorité maximale)
2. Le fichier .env à la racine de seance3/
3. Les valeurs par défaut définies dans cette classe

NB : on calcule un chemin **absolu** vers le fichier .env (et non un chemin
relatif au CWD), pour que la config se charge correctement quelle que soit
la façon dont on lance le code (notebook, script, ou import depuis n'importe où).
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet seance3/ — calculée depuis l'emplacement de ce fichier :
# src/rag_pdf/config.py  →  .parent.parent.parent  =  seance3/
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
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Clé API Groq (obligatoire si llm_provider='groq'). https://console.groq.com/keys",
    )
    cerebras_api_key: Optional[str] = Field(
        default=None,
        description="Clé API Cerebras (obligatoire si llm_provider='cerebras'). https://cloud.cerebras.ai/",
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

    @property
    def evaluation_dir(self) -> Path:
        return self.project_root / "evaluation"

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
    top_k: int = Field(default=4, description="Nombre de chunks récupérés par requête (post-reranking si activé).")
    top_k_initial: int = Field(
        default=20,
        description="Nombre de chunks remontés AVANT reranking (typiquement 4-5x le top_k final).",
    )
    collection_name: str = Field(default="rag_pdf", description="Nom de la collection Chroma.")

    # ------------------------------------------------------------------
    # LLM — choix du fournisseur + modèles par provider
    # ------------------------------------------------------------------
    llm_provider: Literal["groq", "cerebras"] = Field(
        default="groq",
        description="Provider LLM à utiliser. 'groq' ou 'cerebras'.",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Modèle utilisé chez Groq (si llm_provider='groq').",
    )
    cerebras_model: str = Field(
        default="gpt-oss-120b",
        description=(
            "Modèle utilisé chez Cerebras (si llm_provider='cerebras'). "
            "Modèles dispos varient selon ton compte — pour les lister : "
            "`client.models.list()` via le SDK OpenAI. "
            "Suggestions free tier : 'gpt-oss-120b' (très performant) ou 'zai-glm-4.7'."
        ),
    )
    llm_temperature: float = Field(default=0.0, description="0 = déterministe, 1 = créatif.")

    # ------------------------------------------------------------------
    # NEW — Composants enrichis (séance 3)
    # ------------------------------------------------------------------
    use_query_rewriting: bool = Field(
        default=False,
        description="Si True, reformule la question avant le retrieval (pré-retrieval).",
    )
    use_reranker: bool = Field(
        default=False,
        description="Si True, réordonne les chunks récupérés via un cross-encoder (post-retrieval).",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Modèle cross-encoder pour le reranking (multilingue).",
    )


# Instance unique réutilisée dans tout le projet.
settings = Settings()
