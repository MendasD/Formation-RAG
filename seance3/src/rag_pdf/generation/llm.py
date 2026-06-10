"""
Wrappers de LLM — Interface + implémentations multi-providers.

Providers supportés (sélectionnables via `settings.llm_provider`) :
- **Groq** : rapide (~500 tok/s), gratuit, modèles Llama / Mixtral / Gemma.
- **Cerebras** : ultra-rapide (~2000 tok/s), gratuit (~1M tokens/jour), modèles Llama.

L'interface `BaseLLM` permet de swapper de provider en changeant UNE ligne
dans `.env` (`LLM_PROVIDER=cerebras` ou `LLM_PROVIDER=groq`). Le reste du
package (chaîne RAG, évaluation, app Streamlit) est totalement indépendant
du provider choisi.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel

from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


# Type pour la valeur de provider (utile en annotations et autocomplétion).
LLMProvider = Literal["groq", "cerebras"]


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------

class BaseLLM(ABC):
    """Interface d'un fournisseur de LLM."""

    @property
    @abstractmethod
    def langchain_llm(self) -> BaseChatModel:
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        ...


# ----------------------------------------------------------------------
# Implémentation Groq
# ----------------------------------------------------------------------

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
        max_retries: int = 5,
        timeout: int = 60,
    ):
        from langchain_groq import ChatGroq  # import paresseux

        logger.info(f"[bold cyan]Initialisation du LLM Groq :[/] {model}")
        self._llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_retries=max_retries,  # retry auto sur 429/5xx avec backoff exponentiel
            timeout=timeout,
        )
        self._model = model

    @property
    def langchain_llm(self) -> BaseChatModel:
        return self._llm

    @property
    def model(self) -> str:
        return self._model


# ----------------------------------------------------------------------
# Implémentation Cerebras
# ----------------------------------------------------------------------

class CerebrasLLM(BaseLLM):
    """LLM via l'API Cerebras Cloud.

    Cerebras propose une inférence ~4× plus rapide que Groq sur les mêmes
    modèles Llama, avec un quota gratuit généreux (~1M tokens/jour).

    Implémentation : on passe par `ChatOpenAI` avec `base_url=...cerebras...`,
    car Cerebras expose une API **OpenAI-compatible**. C'est plus stable que
    le wrapper officiel `langchain-cerebras` qui a régulièrement des conflits
    de versions avec `langchain-openai`.

    Args:
        model: Nom du modèle Cerebras. Par défaut Llama 3.3 70B.
            Alternatives : "llama3.1-8b", "llama-4-scout-17b-16e-instruct",
            "qwen-3-32b".
        temperature: 0.0 pour un comportement déterministe.
        api_key: Clé API Cerebras. Si None, lue depuis l'env (CEREBRAS_API_KEY).
    """

    CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(
        self,
        model: str = "llama-3.3-70b",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        timeout: int = 60,
    ):
        from langchain_openai import ChatOpenAI  # import paresseux

        logger.info(f"[bold cyan]Initialisation du LLM Cerebras :[/] {model}")
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=self.CEREBRAS_BASE_URL,
            max_retries=max_retries,  # retry auto sur 429/queue_exceeded avec backoff exponentiel
            timeout=timeout,
        )
        self._model = model

    @property
    def langchain_llm(self) -> BaseChatModel:
        return self._llm

    @property
    def model(self) -> str:
        return self._model


# ----------------------------------------------------------------------
# Factory — sélectionne le provider depuis la config
# ----------------------------------------------------------------------

def make_llm(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseLLM:
    """Construit un LLM selon le provider voulu.

    Si les arguments sont None, on lit la config centralisée.

    Args:
        provider: "groq" ou "cerebras". Si None, lit `settings.llm_provider`.
        model: Nom du modèle. Si None, lit le modèle par défaut du provider.
        temperature: Si None, lit `settings.llm_temperature`.

    Returns:
        Une instance de `BaseLLM` prête à être utilisée par la chaîne RAG.

    Raises:
        ValueError: Si le provider n'est pas reconnu.
    """
    # Import retardé pour éviter une dépendance circulaire avec config.py.
    from rag_pdf.config import settings

    provider = provider or settings.llm_provider
    temperature = temperature if temperature is not None else settings.llm_temperature

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY manquante dans .env — impossible d'utiliser "
                "le provider 'groq'. Obtenez une clé sur https://console.groq.com/keys"
            )
        return GroqLLM(
            model=model or settings.llm_model,
            temperature=temperature,
            api_key=settings.groq_api_key,
        )

    if provider == "cerebras":
        if not settings.cerebras_api_key:
            raise ValueError(
                "CEREBRAS_API_KEY manquante dans .env — impossible d'utiliser "
                "le provider 'cerebras'. Obtenez une clé sur https://cloud.cerebras.ai/"
            )
        return CerebrasLLM(
            model=model or settings.cerebras_model,
            temperature=temperature,
            api_key=settings.cerebras_api_key,
        )

    raise ValueError(
        f"Provider LLM inconnu : {provider!r}. Choix : 'groq' ou 'cerebras'."
    )
