"""
Sidebar de l'app Streamlit — Upload PDF, ingestion sélective, paramètres RAG.
"""
from __future__ import annotations

import gc
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from app.resources import get_embedder
from app.state import StateKey, get_rag_config_hash, reset_chat
from rag_pdf.config import settings
from rag_pdf.indexing.vector_store import ChromaVectorStore
from rag_pdf.ingestion.loader import PDFLoader
from rag_pdf.ingestion.splitter import Splitter
from rag_pdf.utils.page_parser import PageRangeError, parse_page_ranges
from rag_pdf.utils.pdf_renderer import get_pdf_page_count


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@contextmanager
def _timed_step(label_pending: str):
    """Context manager qui affiche un step dans st.status avec un timer subtil.

    Usage:
        with _timed_step("Chargement du PDF") as step:
            documents = loader.load(...)
            step["done"] = f"Chargement du PDF ({len(documents)} pages)"
    """
    placeholder = st.empty()
    placeholder.markdown(f"⏳ {label_pending}…")
    t0 = time.perf_counter()
    state: dict[str, str] = {}
    try:
        yield state
        dt = time.perf_counter() - t0
        final = state.get("done", label_pending)
        placeholder.markdown(
            f"✓ {final} <span style='color:#888; font-size:0.85em;'>· {dt:.1f}s</span>",
            unsafe_allow_html=True,
        )
    except Exception:
        dt = time.perf_counter() - t0
        placeholder.markdown(
            f"❌ {label_pending} <span style='color:#888;'>· échec après {dt:.1f}s</span>",
            unsafe_allow_html=True,
        )
        raise


def _safe_remove_persist_dir() -> None:
    """Supprime chroma_db/ de façon robuste, même si Chroma a un handle ouvert (Windows)."""
    persist = settings.chroma_persist_dir
    if not persist.exists():
        return

    # 1. Libère tous les RAG en cache (qui pourraient tenir des handles sur chroma_db).
    if StateKey.RAG_CHAIN in st.session_state and st.session_state[StateKey.RAG_CHAIN] is not None:
        st.session_state[StateKey.RAG_CHAIN] = None
    gc.collect()

    # 2. Plusieurs tentatives (Windows file locks parfois lents à se libérer).
    for _ in range(3):
        try:
            shutil.rmtree(persist)
            return
        except (PermissionError, OSError):
            time.sleep(0.4)
            gc.collect()

    # 3. Dernier recours : ignore_errors (peut laisser quelques fichiers résiduels, sans gravité
    # car add_documents va recréer la collection par-dessus).
    shutil.rmtree(persist, ignore_errors=True)


def render_sidebar() -> None:
    """Rendu complet de la sidebar."""
    st.markdown("## ⚙️ Configuration")

    _section_existing_collection()  # ← NOUVEAU : détection d'une collection préexistante
    _section_pdf_upload()
    st.divider()
    _section_ingestion_settings()
    st.divider()
    _section_rag_settings()
    st.divider()
    _section_actions()


# ----------------------------------------------------------------------
# Section 0 : récupération d'une collection préexistante (ingest CLI)
# ----------------------------------------------------------------------

def _has_existing_collection() -> bool:
    """Vérifie rapidement (sans charger l'embedder) si chroma_db/ a déjà des données."""
    sqlite = settings.chroma_persist_dir / "chroma.sqlite3"
    return sqlite.exists() and sqlite.stat().st_size > 1024  # > 1 Ko = a du contenu


def _section_existing_collection() -> None:
    """Si une base Chroma existe déjà mais que la session est neuve, propose de la réutiliser."""
    if st.session_state[StateKey.COLLECTION_READY]:
        return
    if not _has_existing_collection():
        return

    # Tente de récupérer le PDF associé (celui le plus récent dans data/pdfs/)
    pdfs = sorted(settings.data_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if settings.data_dir.exists() else []

    with st.container():
        st.success("💾 **Collection existante détectée**")
        st.caption(
            "Une base vectorielle est déjà présente dans `chroma_db/` "
            "(probablement créée via `scripts.ingest`)."
        )
        if st.button(
            "🔌 Brancher l'app sur cette collection",
            use_container_width=True,
            type="primary",
        ):
            st.session_state[StateKey.COLLECTION_READY] = True
            st.session_state[StateKey.RAG_CHAIN] = None
            # Associe le PDF le plus récent (pour l'affichage des sources), si dispo.
            if pdfs:
                st.session_state[StateKey.PDF_PATH] = pdfs[0]
                st.session_state[StateKey.PDF_NAME] = pdfs[0].name
                st.session_state[StateKey.PDF_PAGE_COUNT] = get_pdf_page_count(pdfs[0])
            reset_chat()
            st.rerun()
        st.divider()


# ----------------------------------------------------------------------
# Section 1 : upload PDF
# ----------------------------------------------------------------------

def _section_pdf_upload() -> None:
    st.markdown("### 📄 Document")

    # Option 1 : uploader un PDF.
    uploaded = st.file_uploader(
        "Glisser-déposer un PDF",
        type=["pdf"],
        help="Le PDF sera sauvegardé dans data/pdfs/ puis indexé.",
    )

    if uploaded is not None:
        # Sauvegarde dans data/pdfs/
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        target_path = settings.data_dir / uploaded.name
        if (
            st.session_state[StateKey.PDF_PATH] is None
            or st.session_state[StateKey.PDF_PATH] != target_path
        ):
            target_path.write_bytes(uploaded.getvalue())
            st.session_state[StateKey.PDF_PATH] = target_path
            st.session_state[StateKey.PDF_NAME] = uploaded.name
            st.session_state[StateKey.PDF_PAGE_COUNT] = get_pdf_page_count(target_path)
            st.session_state[StateKey.COLLECTION_READY] = False
            st.success(
                f"✓ PDF chargé : **{uploaded.name}** "
                f"({st.session_state[StateKey.PDF_PAGE_COUNT]} pages)"
            )

    # Option 2 : PDF déjà présent dans data/pdfs/
    elif st.session_state[StateKey.PDF_PATH] is None:
        existing_pdfs = list(settings.data_dir.glob("*.pdf")) if settings.data_dir.exists() else []
        if existing_pdfs:
            choice = st.selectbox(
                "Ou choisir un PDF existant :",
                ["—"] + [p.name for p in existing_pdfs],
            )
            if choice and choice != "—":
                target_path = settings.data_dir / choice
                st.session_state[StateKey.PDF_PATH] = target_path
                st.session_state[StateKey.PDF_NAME] = choice
                st.session_state[StateKey.PDF_PAGE_COUNT] = get_pdf_page_count(target_path)
                st.session_state[StateKey.COLLECTION_READY] = False
                st.rerun()

    # Affichage du PDF actuel
    if st.session_state[StateKey.PDF_PATH] is not None:
        st.caption(
            f"📌 **{st.session_state[StateKey.PDF_NAME]}** — "
            f"{st.session_state[StateKey.PDF_PAGE_COUNT]} pages"
        )


# ----------------------------------------------------------------------
# Section 2 : sélection des pages à ingérer + lancement
# ----------------------------------------------------------------------

def _section_ingestion_settings() -> None:
    st.markdown("### 📥 Ingestion")

    if st.session_state[StateKey.PDF_PATH] is None:
        st.caption("Charge d'abord un PDF.")
        return

    total = st.session_state[StateKey.PDF_PAGE_COUNT]
    pages_expr = st.text_input(
        "Pages à indexer",
        value="",
        placeholder=f"ex: 1-{total}, ou 1-5, 10, 15-20",
        help=(
            "Notation compacte avec intervalles disjoints.\n"
            "- Vide = tout le PDF\n"
            f"- `1-{total}` = tout aussi\n"
            "- `1-5, 10, 15-20` = pages 1 à 5, 10, et 15 à 20"
        ),
    )

    # Aperçu temps réel des pages parsées
    try:
        pages_preview = parse_page_ranges(pages_expr, max_page=total) if pages_expr else None
        if pages_preview is not None:
            st.caption(
                f"→ {len(pages_preview)} pages sur {total} sélectionnées"
            )
        elif pages_expr == "":
            st.caption(f"→ Tout le PDF ({total} pages)")
    except PageRangeError as exc:
        st.error(f"⚠️ {exc}")
        pages_preview = None

    chunk_size = st.number_input(
        "Chunk size",
        min_value=200,
        max_value=2000,
        value=settings.chunk_size,
        step=100,
        help="Taille cible d'un chunk en caractères. 800 = bon défaut.",
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=settings.chunk_overlap,
        step=20,
        help="Chevauchement entre chunks consécutifs.",
    )

    if st.button(
        "🚀 Lancer l'ingestion",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state[StateKey.PDF_PATH] is None),
    ):
        _do_ingest(pages_preview, chunk_size, chunk_overlap)


def _do_ingest(pages: list[int] | None, chunk_size: int, chunk_overlap: int) -> None:
    """Exécute l'ingestion étape par étape — version sans st.status (plus stable)."""
    pdf_path = st.session_state[StateKey.PDF_PATH]
    success = False
    error_msg: str | None = None
    n_chunks_final = 0
    total_start = time.perf_counter()

    # Cadre visuel simple : un container avec une barre de progression + log
    container = st.container()
    with container:
        st.markdown("**🚀 Ingestion en cours…**")
        progress = st.progress(0.0)
        log_area = st.empty()
        log_lines: list[str] = []

        def _log(line: str) -> None:
            """Ajoute une ligne au log visible + au terminal."""
            log_lines.append(line)
            log_area.markdown("  \n".join(log_lines), unsafe_allow_html=True)
            print(f"[ingest] {line}", file=sys.stderr, flush=True)

        def _step(pct: float, label: str, t0: float, suffix: str = "") -> None:
            """Termine une étape : update barre + log avec temps."""
            dt = time.perf_counter() - t0
            progress.progress(pct, text=f"{label} · {dt:.1f}s")
            _log(f"✓ {label}{(' · ' + suffix) if suffix else ''} <span style='color:#888'>· {dt:.1f}s</span>")

        try:
            # 1) Chargement du PDF
            t0 = time.perf_counter()
            documents = PDFLoader().load(pdf_path, pages=pages)
            _step(0.10, "Chargement du PDF", t0, f"{len(documents)} pages")

            # 2) Chunking
            t0 = time.perf_counter()
            splitter = Splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split(documents)
            _step(0.25, "Découpage en chunks", t0,
                  f"{len(chunks)} chunks (cible {chunk_size}, overlap {chunk_overlap})")

            # 3) Embedder — récupéré depuis le cache global (chargé UNE seule fois
            # pour toute la durée de l'app, peu importe le nombre d'ingestions).
            t0 = time.perf_counter()
            embedder = get_embedder()
            _step(0.45, "Modèle d'embeddings prêt", t0, settings.embedding_model.split("/")[-1])

            # 4) Reset propre de la collection précédente
            t0 = time.perf_counter()
            _safe_remove_persist_dir()
            _step(0.55, "Collection précédente nettoyée", t0)

            # 5) Indexation
            t0 = time.perf_counter()
            vector_store = ChromaVectorStore(
                embedder=embedder,
                persist_directory=settings.chroma_persist_dir,
                collection_name=settings.collection_name,
            )
            vector_store.add_documents(chunks)
            n_chunks_final = vector_store.count()
            _step(1.00, "Indexation dans ChromaDB", t0, f"{n_chunks_final} chunks")

            total = time.perf_counter() - total_start
            progress.progress(1.0, text=f"✓ Ingestion réussie — {n_chunks_final} chunks · {total:.1f}s")
            _log(f"<b style='color:#008c5a'>🎉 Terminé en {total:.1f}s — {n_chunks_final} chunks dans Chroma.</b>")
            success = True

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            _log(f"<b style='color:#c83232'>❌ Erreur : {exc}</b>")
            print("[ingest] FAILED — stacktrace ci-dessous :", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

    # ── Hors du container : mise à jour du state ──────────────────────
    if success:
        st.session_state[StateKey.COLLECTION_READY] = True
        st.session_state[StateKey.INDEXED_PAGES] = pages or list(
            range(1, st.session_state[StateKey.PDF_PAGE_COUNT] + 1)
        )
        st.session_state[StateKey.RAG_CHAIN] = None
        st.session_state[StateKey.INGEST_ERROR] = None
        reset_chat()
        # Pas de st.rerun() : le routing dans streamlit_app.py va voir
        # COLLECTION_READY=True et basculer sur le chat au rerun naturel suivant.
    else:
        st.session_state[StateKey.INGEST_ERROR] = error_msg


# ----------------------------------------------------------------------
# Section 3 : paramètres du RAG (modifiables à la volée)
# ----------------------------------------------------------------------

def _section_rag_settings() -> None:
    st.markdown("### 🛠️ Paramètres RAG")

    st.session_state[StateKey.TOP_K] = st.slider(
        "Top-k (nombre de chunks)",
        min_value=2,
        max_value=15,
        value=st.session_state[StateKey.TOP_K],
        help="Nombre de chunks récupérés (et passés au LLM).",
    )

    st.session_state[StateKey.USE_QUERY_REWRITING] = st.toggle(
        "🔄 Reformulation de question",
        value=st.session_state[StateKey.USE_QUERY_REWRITING],
        help="Pré-retrieval : le LLM reformule la question avant la recherche. Coût : +1 appel LLM.",
    )

    st.session_state[StateKey.USE_RERANKER] = st.toggle(
        "🎯 Reranker (cross-encoder)",
        value=st.session_state[StateKey.USE_RERANKER],
        help=(
            "Post-retrieval : un cross-encoder précis réordonne les chunks. "
            "Coût : ~200-500 ms par requête."
        ),
    )

    st.session_state[StateKey.SHOW_SOURCES] = st.toggle(
        "📑 Afficher les sources",
        value=st.session_state[StateKey.SHOW_SOURCES],
        help="Affiche le panneau de droite avec les chunks sources et le rendu de la page PDF.",
    )


# ----------------------------------------------------------------------
# Section 4 : actions diverses
# ----------------------------------------------------------------------

def _section_actions() -> None:
    st.markdown("### 🎬 Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Vider chat", use_container_width=True):
            reset_chat()
            st.rerun()

    with col2:
        if st.button("♻️ Reset DB", use_container_width=True, type="secondary"):
            if settings.chroma_persist_dir.exists():
                shutil.rmtree(settings.chroma_persist_dir)
            st.session_state[StateKey.COLLECTION_READY] = False
            st.session_state[StateKey.RAG_CHAIN] = None
            reset_chat()
            st.toast("Collection vectorielle supprimée.", icon="♻️")
            st.rerun()

    if st.session_state[StateKey.COLLECTION_READY]:
        with st.expander("ℹ️ Statut de la collection"):
            indexed = st.session_state[StateKey.INDEXED_PAGES]
            from rag_pdf.utils.page_parser import format_pages_compact
            st.caption(
                f"**Document :** {st.session_state[StateKey.PDF_NAME]}\n\n"
                f"**Pages indexées :** {format_pages_compact(indexed) or '—'}\n\n"
                f"**Total :** {len(indexed)} pages"
            )
