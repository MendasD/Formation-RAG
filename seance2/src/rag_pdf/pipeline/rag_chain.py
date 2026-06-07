"""
Pipeline RAG complet — assemblage des composants en une chaîne.

C'est le coeur du système : on prend une question, on retrouve les chunks
pertinents, on construit un prompt avec ces chunks comme contexte,
on l'envoie au LLM, et on emballe le tout dans un `RAGAnswer` typé.

L'utilisation de LCEL (LangChain Expression Language) avec l'opérateur `|`
rend la composition très lisible : c'est notre "expression mathématique"
du RAG.
"""
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag_pdf.generation.llm import BaseLLM
from rag_pdf.generation.prompts import build_rag_prompt
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
    """Orchestrateur du pipeline RAG : retrieval + génération.

    Args:
        retriever: Composant de recherche (Naive en séance 2,
            potentiellement enrichi en séance 3).
        llm: LLM de génération.
    """

    def __init__(self, retriever: Retriever, llm: BaseLLM):
        self.retriever = retriever
        self.llm = llm
        self.prompt = build_rag_prompt()

        # Sous-chaîne LCEL réutilisable : (context, question) -> réponse texte.
        # On la construit une fois, on l'invoque à chaque question.
        self._generate = self.prompt | self.llm.langchain_llm | StrOutputParser()

    def invoke(self, question: str) -> RAGAnswer:
        """Exécute le pipeline complet sur une question.

        Étapes :
        1. Retrieval : on récupère les top-k chunks.
        2. Formatage : on construit le bloc de contexte.
        3. Génération : on demande au LLM de répondre.
        4. Emballage : on retourne un `RAGAnswer` typé avec les sources.
        """
        logger.info(f"[bold]Question :[/] {question}")

        # 1. Retrieval
        docs = self.retriever.retrieve(question)
        logger.info(
            f"[dim]→ {len(docs)} chunks récupérés "
            f"(pages : {sorted({d.metadata.get('page', '?') for d in docs})})[/]"
        )

        # 2. Formatage du contexte
        context = format_context(docs)

        # 3. Génération
        answer_text = self._generate.invoke(
            {"context": context, "question": question}
        )

        # 4. Construction de la réponse typée
        sources = [
            SourceChunk(
                page=doc.metadata.get("page", 0),
                content=doc.page_content,
            )
            for doc in docs
        ]

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
        )
    
    def stream(self, question: str) -> Iterator[str]:
        """Exécute le pipeline en mode streaming : yield les tokens un par un.

        Idéal pour les UIs (chat web, CLI interactive) : on affiche la réponse
        au fur et à mesure qu'elle est générée, plutôt qu'attendre la fin.

        Args:
            question: La question de l'utilisateur.

        Yields:
            Des chunks de texte (typiquement quelques tokens chacun).

        Notes:
            Les sources ne sont **pas** yieldées ici — un générateur ne renvoie
            qu'un seul type pour rester simple côté caller. Après le stream,
            elles sont disponibles via `self.last_sources` (mis à jour à chaque
            appel). Sinon, utilise `.invoke()` qui renvoie un `RAGAnswer`
            complet (réponse + sources).
        """
        logger.info(f"[bold]Question (stream) :[/] {question}")

        # 1. Retrieval (non streamable — on récupère tout d'un coup)
        docs = self.retriever.retrieve(question)
        logger.info(
            f"[dim]→ {len(docs)} chunks récupérés "
            f"(pages : {sorted({d.metadata.get('page', '?') for d in docs})})[/]"
        )

        # 2. On stocke les sources pour qu'elles soient consultables APRÈS le stream.
        self.last_sources = [
            SourceChunk(
                page=doc.metadata.get("page", 0),
                content=doc.page_content,
            )
            for doc in docs
        ]

        # 3. Formatage du contexte + génération streamée
        context = format_context(docs)
        yield from self._generate.stream(
            {"context": context, "question": question}
        )

