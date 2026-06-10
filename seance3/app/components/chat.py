"""
Composant Chat — affichage de l'historique + nouvelle question avec streaming.
"""
from __future__ import annotations

import time

import streamlit as st

from app.resources import build_cached_rag_chain
from app.state import StateKey, add_message, get_messages, get_rag_config_hash
from rag_pdf.schemas import SourceChunk


def _get_or_build_rag():
    """Construit le RAG (ou récupère celui mis en cache) avec la config courante.

    Utilise `build_cached_rag_chain()` qui réutilise les ressources lourdes
    (embedder, LLM, reranker) en cache `st.cache_resource`, donc seul
    l'assemblage final est refait à chaque changement de config — ce qui
    prend < 100 ms au lieu de plusieurs secondes.
    """
    current_hash = get_rag_config_hash()
    if (
        st.session_state[StateKey.RAG_CHAIN] is None
        or st.session_state[StateKey.RAG_CONFIG_HASH] != current_hash
    ):
        st.session_state[StateKey.RAG_CHAIN] = build_cached_rag_chain(
            use_query_rewriting=st.session_state[StateKey.USE_QUERY_REWRITING],
            use_reranker=st.session_state[StateKey.USE_RERANKER],
            top_k=st.session_state[StateKey.TOP_K],
        )
        st.session_state[StateKey.RAG_CONFIG_HASH] = current_hash
    return st.session_state[StateKey.RAG_CHAIN]


def render_chat() -> None:
    """Affiche l'historique de chat + l'input pour une nouvelle question."""
    st.markdown("### 💬 Conversation")

    # Bannière de config courante (info légère)
    badges = []
    if st.session_state[StateKey.USE_QUERY_REWRITING]:
        badges.append("🔄 Rewrite")
    if st.session_state[StateKey.USE_RERANKER]:
        badges.append("🎯 Rerank")
    badges.append(f"top-k={st.session_state[StateKey.TOP_K]}")
    st.caption(" · ".join(badges))

    # Historique
    for msg in get_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_message_footer(msg)

    # Input
    if prompt := st.chat_input("Pose une question sur le document…"):
        _handle_new_question(prompt)


def _render_message_footer(msg: dict) -> None:
    """Footer d'un message assistant : pages citées + latence + rewrite éventuelle."""
    parts = []
    if msg.get("sources"):
        unique_pages = sorted({s.page for s in msg["sources"]})
        # Boutons cliquables pour ouvrir la page dans le panneau de droite.
        cols = st.columns([1] * min(len(unique_pages), 8) + [1])
        for i, page in enumerate(unique_pages[:8]):
            with cols[i]:
                if st.button(f"📄 p.{page}", key=f"page_btn_{msg['ts']}_{page}"):
                    st.session_state[StateKey.SELECTED_PAGE] = page
                    st.rerun()

    sub_caption = []
    if msg.get("rewritten"):
        sub_caption.append(f"🔄 *« {msg['rewritten']} »*")
    if msg.get("latency_ms"):
        sub_caption.append(f"⏱️ {msg['latency_ms']:.0f} ms")
    if sub_caption:
        st.caption(" — ".join(sub_caption))


def _handle_new_question(prompt: str) -> None:
    """Traite une nouvelle question avec streaming."""
    # 1. Affiche immédiatement la question (et la stocke).
    add_message("user", prompt, ts=time.time())
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Récupère le RAG (construction si nécessaire avec la config courante).
    rag = _get_or_build_rag()

    # 3. Génère la réponse en streaming.
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        t0 = time.perf_counter()

        try:
            for chunk in rag.stream(prompt):
                full_text += chunk
                placeholder.markdown(full_text + " ▌")
            placeholder.markdown(full_text)
        except Exception as exc:  # noqa: BLE001
            placeholder.error(f"❌ Erreur durant la génération : {exc}")
            return

        latency_ms = (time.perf_counter() - t0) * 1000
        sources: list[SourceChunk] = rag.last_sources
        rewritten = rag.last_rewritten_question

        # Stocke le message complet.
        ts = time.time()
        add_message(
            "assistant",
            full_text,
            sources=sources,
            rewritten=rewritten,
            latency_ms=latency_ms,
            ts=ts,
        )

        # Met en évidence la 1re page source pour l'affichage immédiat dans le panneau.
        if sources:
            st.session_state[StateKey.SELECTED_PAGE] = sources[0].page

        # Affiche le footer (rerun pour que les boutons soient bien indexés)
        st.rerun()
