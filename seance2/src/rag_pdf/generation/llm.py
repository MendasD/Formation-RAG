"""
Wrapper de LLM — Interface + implémentation Groq.

Groq propose une API gratuite ultra-rapide (~500 tokens/sec) pour
plusieurs modèles open-weights (Llama, Mixtral). Idéal pour la formation.

L'interface `BaseLLM` permettra plus tard de swapper vers OpenAI,
Anthropic ou Ollama sans changer la chaîne RAG.
"""
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    """Interface d'un fournisseur de LLM."""

    @property
    @abstractmethod
    def langchain_llm(self) -> BaseChatModel:
        ...


class GroqLLM(BaseLLM):
    """LLM via l'API Groq.

    Args:
        model: Nom du modèle Groq. Par défaut Llama 3.3 70B (rapide et solide).
            Alternatives : "llama-3.1-8b-instant", "mixtral-8x7b-32768".
        temperature: 0.0 pour un comportement déterministe (recommandé en RAG).
        api_key: Clé API Groq. Si None, lue depuis l'env (GROQ_API_KEY).
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        logger.info(f"[bold cyan]Initialisation du LLM Groq :[/] {model}")
        self._llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        self._model = model

    @property
    def langchain_llm(self) -> BaseChatModel:
        return self._llm

    @property
    def model(self) -> str:
        return self._model
