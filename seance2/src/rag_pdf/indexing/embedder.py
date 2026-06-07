"""
Modèles d'embeddings — Interface + implémentation HuggingFace.

On définit une interface abstraite `BaseEmbedder` pour pouvoir swapper
facilement entre fournisseurs (HuggingFace, OpenAI, Cohere…) sans
toucher au reste du package. C'est le pattern Stratégie.
"""
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Interface d'un fournisseur d'embeddings.

    Toute implémentation doit exposer :
    - `langchain_embeddings` : l'objet `Embeddings` LangChain sous-jacent,
      utilisable directement par les vector stores et chaînes LangChain.
    - `embed(text)` : helper de commodité pour embedder un seul texte.
    """

    @property
    @abstractmethod
    def langchain_embeddings(self) -> Embeddings:
        ...

    def embed(self, text: str) -> list[float]:
        return self.langchain_embeddings.embed_query(text)


class HFEmbedder(BaseEmbedder):
    """Embeddings via un modèle HuggingFace téléchargé localement.

    Le modèle par défaut (`intfloat/multilingual-e5-base`) est multilingue,
    supporte bien le français, et fait ~1 Go. Au premier appel il est
    téléchargé puis mis en cache (`~/.cache/huggingface/`).
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"[bold cyan]Chargement du modèle d'embeddings :[/] {model_name}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            # normalize_embeddings=True → vecteurs unitaires
            # → la similarité cosinus devient un simple produit scalaire (plus rapide).
            encode_kwargs={"normalize_embeddings": True},
        )
        self._model_name = model_name

    @property
    def langchain_embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def model_name(self) -> str:
        return self._model_name
