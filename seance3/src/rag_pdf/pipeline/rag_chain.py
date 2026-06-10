"""
Pipeline RAG complet — assemblage des composants en une chaîne.

Évolution séance 3 :
- Le pipeline supporte maintenant la reformulation de question (pré-retrieval)
  et le reranking (post-retrieval, dans le retriever).
- `RAGAnswer` inclut éventuellement la question reformulée et la latence.

L'utilisation de LCEL (LangChain Expression Language) avec l'opérateur `|`
reste centrale pour la sous-chaîne de génération. Le reste est impératif
pour pouvoir capturer toutes les valeurs intermédiaires (sources, reformulation).
"""
import time
from typing import Iterator, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag_pdf.generation.llm import BaseLLM
from rag_pdf.generation.prompts import build_rag_prompt
from rag_pdf.retrieval.query_rewriter import BaseQueryRewriter, NoOpRewriter
from rag_pdf.retrieval.retriever import Retriever
from rag_pdf.schemas import RAGAnswer, SourceChunk
from rag_pdf.utils.logging import get_logger

logger = get_logger(__name__)


def format_context(docs: list[Document]) -> str:
    """Formate les chunks récupérés en un bloc de contexte lisible par le LLM.

    Chaque chunk est précédé d'une balise [Page X] : c'est ce qui permet
    au LLM de citer correctement ses sources.
    """
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGChain:
    """Orchestrateur du pipeline RAG : (reformulation →) retrieval (→ reranking) → génération.

    Args:
        retriever: Composant de recherche (incluant éventuellement un reranker).
        llm: LLM de génération.
        query_rewriter: Reformulateur de question (NoOp par défaut, équivalent baseline).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        query_rewriter: Optional[BaseQueryRewriter] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.query_rewriter = query_rewriter or NoOpRewriter()
        self.prompt = build_rag_prompt()
        self._has_rewriter = not isinstance(self.query_rewriter, NoOpRewriter)

        # Sous-chaîne LCEL réutilisable : (context, question) -> réponse texte.
        self._generate = self.prompt | self.llm.langchain_llm | StrOutputParser()

        # Mémorisation post-stream (pour récupérer les sources après un .stream()).
        self.last_sources: list[SourceChunk] = []
        self.last_rewritten_question: Optional[str] = None

    # ------------------------------------------------------------------
    # Pipeline interne partagé entre invoke() et stream()
    # ------------------------------------------------------------------

    def _prepare(self, question: str) -> tuple[str, str, list[Document]]:
        """Effectue la reformulation (optionnelle) + le retrieval.

        Returns:
            (search_query, context, docs)
        """
        # 1. Reformulation éventuelle.
        search_query = self.query_rewriter.rewrite(question) if self._has_rewriter else question

        # 2. Retrieval (avec reranking interne si activé).
        docs = self.retriever.retrieve(search_query)
        logger.info(
            f"[dim]→ {len(docs)} chunks récupérés "
            f"(pages : {sorted({d.metadata.get('page', '?') for d in docs})})[/]"
        )

        # 3. Formatage du contexte pour le prompt.
        context = format_context(docs)

        return search_query, context, docs

    def _to_source_chunks(self, docs: list[Document]) -> list[SourceChunk]:
        """Convertit les Document LangChain en SourceChunk Pydantic."""
        return [
            SourceChunk(
                page=doc.metadata.get("page", 0),
                content=doc.page_content,
                score=doc.metadata.get("score"),
                rerank_score=doc.metadata.get("rerank_score"),
            )
            for doc in docs
        ]

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def invoke(self, question: str) -> RAGAnswer:
        """Exécute le pipeline complet sur une question et retourne un RAGAnswer typé."""
        logger.info(f"[bold]Question :[/] {question}")
        t0 = time.perf_counter()

        search_query, context, docs = self._prepare(question)

        answer_text = self._generate.invoke(
            {"context": context, "question": question}
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return RAGAnswer(
            question=question,
            rewritten_question=search_query if self._has_rewriter else None,
            answer=answer_text,
            sources=self._to_source_chunks(docs),
            latency_ms=latency_ms,
        )

    def stream(self, question: str) -> Iterator[str]:
        """Exécute le pipeline en mode streaming : yield les tokens un par un.

        Les sources et la reformulation sont disponibles après le stream via :
            self.last_sources
            self.last_rewritten_question
        """
        logger.info(f"[bold]Question (stream) :[/] {question}")

        search_query, context, docs = self._prepare(question)

        # On stocke pour consultation post-stream.
        self.last_sources = self._to_source_chunks(docs)
        self.last_rewritten_question = search_query if self._has_rewriter else None

        yield from self._generate.stream(
            {"context": context, "question": question}
        )
