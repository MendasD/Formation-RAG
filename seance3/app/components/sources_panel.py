"""
Panneau de sources — affichage des chunks récupérés + rendu de la page PDF.

Affiché à droite du chat quand `SHOW_SOURCES = True`.
"""
from __future__ import annotations

import streamlit as st

from app.state import StateKey, get_last_sources


def render_sources_panel() -> None:
    """Panneau de droite : sources du dernier message + rendu de la page PDF."""
    st.markdown("### 📑 Sources")

    sources = get_last_sources()
    if not sources:
        st.info("Pose une question pour voir les sources s'afficher ici.")
        return

    # Tabs : Liste des chunks · Rendu PDF
    tab_chunks, tab_pdf = st.tabs(["🧩 Chunks", "📄 Page PDF"])

    # ── Tab 1 : liste des chunks récupérés ──────────────────────────
    with tab_chunks:
        unique_pages = sorted({s.page for s in sources})
        st.caption(
            f"**{len(sources)} chunks** récupérés sur **{len(unique_pages)} pages** : "
            f"{', '.join(f'p.{p}' for p in unique_pages)}"
        )

        for i, src in enumerate(sources, 1):
            with st.container():
                # En-tête : numéro + page + scores
                head_cols = st.columns([3, 2])
                with head_cols[0]:
                    st.markdown(
                        f'<span class="page-badge">Page {src.page}</span> '
                        f'<span style="color:#666; font-size: 0.9em;">chunk {i}</span>',
                        unsafe_allow_html=True,
                    )
                with head_cols[1]:
                    if src.rerank_score is not None:
                        st.markdown(
                            f"<div style='text-align: right; color: #008c5a; font-size: 0.85em;'>"
                            f"rerank: <b>{src.rerank_score:.2f}</b></div>",
                            unsafe_allow_html=True,
                        )
                    elif src.score is not None:
                        st.markdown(
                            f"<div style='text-align: right; color: #666; font-size: 0.85em;'>"
                            f"distance: {src.score:.3f}</div>",
                            unsafe_allow_html=True,
                        )

                # Aperçu du chunk (rétractable si long)
                excerpt = src.content.strip()
                if len(excerpt) > 350:
                    with st.expander(excerpt[:350].rstrip() + "…", expanded=False):
                        st.write(excerpt)
                else:
                    st.write(excerpt)

                # Bouton "Voir cette page"
                if st.button(
                    "→ Voir la page",
                    key=f"src_view_{i}_{src.page}",
                    help="Afficher l'image de cette page dans l'onglet PDF.",
                    use_container_width=False,
                ):
                    st.session_state[StateKey.SELECTED_PAGE] = src.page
                    st.rerun()

                st.divider()

    # ── Tab 2 : rendu de la page PDF sélectionnée ───────────────────
    with tab_pdf:
        _render_pdf_page()


def _render_pdf_page() -> None:
    """Affiche la page PDF actuellement sélectionnée."""
    pdf_path = st.session_state.get(StateKey.PDF_PATH)
    selected = st.session_state.get(StateKey.SELECTED_PAGE)
    total = st.session_state.get(StateKey.PDF_PAGE_COUNT, 0)

    if not pdf_path or not total:
        st.info("Aucun PDF chargé.")
        return

    # Si aucune page sélectionnée, prend la 1re page source.
    if selected is None:
        sources = get_last_sources()
        if sources:
            selected = sources[0].page
            st.session_state[StateKey.SELECTED_PAGE] = selected
        else:
            st.info("Clique sur « Voir la page » sur un chunk pour afficher la page PDF.")
            return

    # Navigation page par page
    nav = st.columns([1, 3, 1])
    with nav[0]:
        if st.button("◀", use_container_width=True, disabled=selected <= 1, key="pdf_prev"):
            st.session_state[StateKey.SELECTED_PAGE] = max(1, selected - 1)
            st.rerun()
    with nav[1]:
        new_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total,
            value=selected,
            step=1,
            label_visibility="collapsed",
            key="pdf_page_input",
        )
        if new_page != selected:
            st.session_state[StateKey.SELECTED_PAGE] = int(new_page)
            st.rerun()
    with nav[2]:
        if st.button("▶", use_container_width=True, disabled=selected >= total, key="pdf_next"):
            st.session_state[StateKey.SELECTED_PAGE] = min(total, selected + 1)
            st.rerun()

    st.caption(f"Page {selected} / {total}")

    # Rendu de la page en PNG (cached côté pdf_renderer).
    from rag_pdf.utils.pdf_renderer import render_page_to_png

    try:
        png_bytes = render_page_to_png(str(pdf_path), selected, dpi=150)
        st.image(png_bytes, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible d'afficher la page : {exc}")
