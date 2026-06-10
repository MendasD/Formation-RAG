"""
Reformulation de question (pré-retrieval).

Une question d'utilisateur est souvent vague, contextuelle, ou utilise un
vocabulaire éloigné du corpus indexé. Reformuler la question via un LLM
avant la recherche améliore généralement le retrieval — c'est ce que
le cours appelle « Advanced RAG > pré-retrieval ».

On définit une interface abstraite + plusieurs implémentations :
- SimpleRewriter : un appel LLM qui reformule en une question plus claire
- HyDERewriter   : génère une réponse hypothétique, qu'on embeddera ensuite
- NoOpRewriter   : pas de reformulation (équivalent baseline)
"""
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from rag_pdf.generation.prompts import build_hyde_prompt, build_query_rewrite_prompt
from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


class BaseQueryRewriter(ABC):
    """Interface d'un reformulateur de question."""

    @abstractmethod
    def rewrite(self, question: str) -> str:
        """Renvoie une nouvelle formulation à utiliser pour le retrieval.

        Note : la question RÉPONDUE par le LLM final reste l'originale.
        Seule la requête envoyée au vector store change.
        """
        ...


class NoOpRewriter(BaseQueryRewriter):
    """Ne reformule rien — utilisé comme baseline équivalente au Naive RAG."""

    def rewrite(self, question: str) -> str:
        return question


class SimpleRewriter(BaseQueryRewriter):
    """Reformulation via un seul appel LLM avec un prompt dédié.

    Coût : +1 appel LLM par question. Bénéfice : meilleure pertinence
    sur les questions vagues, mal formulées, ou contextuelles.
    """

    def __init__(self, llm: BaseChatModel):
        self._chain = build_query_rewrite_prompt() | llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        rewritten = self._chain.invoke({"question": question}).strip()
        logger.info(f"[dim]Reformulation :[/] {question!r} → {rewritten!r}")
        return rewritten


class HyDERewriter(BaseQueryRewriter):
    """HyDE (Hypothetical Document Embeddings).

    Au lieu d'embedder la question, on embedde une RÉPONSE hypothétique
    générée par le LLM. Contre-intuitif mais souvent plus efficace,
    car une réponse hypothétique ressemble plus aux chunks réels qu'une question.
    """

    def __init__(self, llm: BaseChatModel):
        self._chain = build_hyde_prompt() | llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        hypothetical = self._chain.invoke({"question": question}).strip()
        logger.info(f"[dim]HyDE :[/] {question!r} → {hypothetical[:80]!r}…")
        return hypothetical
