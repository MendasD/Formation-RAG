"""
Gestion centralisée du `st.session_state` de l'app Streamlit.

Tout ce qui doit persister entre les reruns vit ici. Streamlit re-exécute
le script entier à chaque interaction utilisateur, donc tout état doit
être stocké explicitement dans `st.session_state`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from rag_pdf.schemas import SourceChunk


# ----------------------------------------------------------------------
# Clés du session_state (centralisées pour éviter les fautes de frappe)
# ----------------------------------------------------------------------

class StateKey:
    # Indexation
    PDF_PATH = "pdf_path"
    PDF_NAME = "pdf_name"
    PDF_PAGE_COUNT = "pdf_page_count"
    INDEXED_PAGES = "indexed_pages"
    COLLECTION_READY = "collection_ready"
    INGEST_ERROR = "ingest_error"

    # RAG pipeline
    RAG_CHAIN = "rag_chain"
    RAG_CONFIG_HASH = "rag_config_hash"  # pour détecter les changements de config

    # Settings (modifiables dans la sidebar)
    USE_QUERY_REWRITING = "use_query_rewriting"
    USE_RERANKER = "use_reranker"
    TOP_K = "top_k"

    # Chat
    MESSAGES = "messages"  # list[dict{role, content, sources?, rewritten?}]
    SHOW_SOURCES = "show_sources"
    SELECTED_PAGE = "selected_page"  # page PDF à afficher dans le panneau de droite


def initialize_state() -> None:
    """Initialise toutes les clés du session_state avec des valeurs par défaut."""
    defaults: dict[str, Any] = {
        StateKey.PDF_PATH: None,
        StateKey.PDF_NAME: None,
        StateKey.PDF_PAGE_COUNT: 0,
        StateKey.INDEXED_PAGES: [],
        StateKey.COLLECTION_READY: False,
        StateKey.INGEST_ERROR: None,
        StateKey.RAG_CHAIN: None,
        StateKey.RAG_CONFIG_HASH: None,
        StateKey.USE_QUERY_REWRITING: False,
        StateKey.USE_RERANKER: False,
        StateKey.TOP_K: 4,
        StateKey.MESSAGES: [],
        StateKey.SHOW_SOURCES: True,
        StateKey.SELECTED_PAGE: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_rag_config_hash() -> tuple[bool, bool, int]:
    """Hash léger des paramètres qui requièrent une reconstruction du RAG."""
    return (
        st.session_state[StateKey.USE_QUERY_REWRITING],
        st.session_state[StateKey.USE_RERANKER],
        st.session_state[StateKey.TOP_K],
    )


def reset_chat() -> None:
    """Vide l'historique de chat."""
    st.session_state[StateKey.MESSAGES] = []
    st.session_state[StateKey.SELECTED_PAGE] = None


def add_message(role: str, content: str, **extra: Any) -> None:
    """Ajoute un message à l'historique de chat."""
    msg: dict[str, Any] = {"role": role, "content": content}
    msg.update(extra)
    st.session_state[StateKey.MESSAGES].append(msg)


def get_messages() -> list[dict[str, Any]]:
    return st.session_state[StateKey.MESSAGES]


def get_last_sources() -> list[SourceChunk]:
    """Retourne les sources du dernier message assistant."""
    for msg in reversed(get_messages()):
        if msg["role"] == "assistant" and "sources" in msg:
            return msg["sources"]
    return []
