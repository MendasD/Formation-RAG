"""
Ressources lourdes mises en cache global Streamlit.

Streamlit ré-exécute le script entier à chaque interaction utilisateur.
Sans cache, embedder + LLM + reranker seraient rechargés à chaque rerun,
ce qui prend 15-60 secondes — inacceptable pour une UI.

`st.cache_resource` mémorise une ressource entre tous les reruns ET tous
les utilisateurs (singleton à l'échelle du process Python). Idéal pour
des modèles ML qu'on veut charger UNE seule fois pour toute la session
de l'app.

Architecture :
    get_embedder()      — singleton du modèle d'embedding HF (~1 Go RAM)
    get_llm()           — singleton du client LLM (Groq/Cerebras)
    get_reranker()      — singleton du cross-encoder BGE (~2 Go RAM)
    get_vector_store()  — non caché (peut changer si on réindexe)
"""
from __future__ import annotations

import streamlit as st

from rag_pdf.config import settings
from rag_pdf.generation.llm import BaseLLM, make_llm
from rag_pdf.indexing.embedder import HFEmbedder
from rag_pdf.indexing.vector_store import ChromaVectorStore
from rag_pdf.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    NoOpReranker,
)


@st.cache_resource(show_spinner="🧠 Chargement du modèle d'embeddings…")
def get_embedder() -> HFEmbedder:
    """Modèle d'embeddings — chargé UNE FOIS pour toute la durée de l'app."""
    return HFEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )


@st.cache_resource(show_spinner="🤖 Initialisation du LLM…")
def get_llm() -> BaseLLM:
    """LLM (Groq ou Cerebras selon settings) — singleton."""
    return make_llm()


@st.cache_resource(show_spinner="🎯 Chargement du reranker (premier appel uniquement)…")
def get_reranker() -> BaseReranker:
    """Reranker cross-encoder BGE — chargé UNE FOIS (~30 s la 1re fois)."""
    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.embedding_device,
    )


def get_vector_store() -> ChromaVectorStore:
    """Vector store — NON caché (instanciation rapide, pointe sur le disque)."""
    return ChromaVectorStore(
        embedder=get_embedder(),
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )


# ----------------------------------------------------------------------
# Builder léger qui réutilise les ressources cachées
# ----------------------------------------------------------------------

def build_cached_rag_chain(
    *,
    use_query_rewriting: bool = False,
    use_reranker: bool = False,
    top_k: int = 4,
):
    """Construit un RAGChain en réutilisant les ressources cachées.

    À utiliser DANS L'APP STREAMLIT, à la place de `build_rag_chain()`
    pour éviter de recharger embedder/LLM/reranker à chaque toggle.

    Le seul truc qu'on instancie ici est :
    - Un Retriever (léger, juste un wrapper)
    - Un éventuel SimpleRewriter (juste un prompt + LLM cached)
    - Le RAGChain final (juste un assemblage)
    """
    from rag_pdf.pipeline.rag_chain import RAGChain
    from rag_pdf.retrieval.query_rewriter import (
        NoOpRewriter,
        SimpleRewriter,
    )
    from rag_pdf.retrieval.retriever import Retriever

    # Composants lourds — récupérés depuis le cache.
    embedder = get_embedder()       # noqa: F841 (déjà appelé via vector_store)
    llm = get_llm()
    vector_store = get_vector_store()
    reranker: BaseReranker = get_reranker() if use_reranker else NoOpReranker()

    # Composants légers — créés à la volée.
    retriever = Retriever(
        vector_store=vector_store,
        k=top_k,
        k_initial=settings.top_k_initial,
        reranker=reranker,
    )
    rewriter = SimpleRewriter(llm.langchain_llm) if use_query_rewriting else NoOpRewriter()

    return RAGChain(retriever=retriever, llm=llm, query_rewriter=rewriter)
