"""
App Streamlit — Interface web professionnelle pour le RAG sur PDF.

Lancer avec :
    uv run streamlit run app/streamlit_app.py

Architecture :
    streamlit_app.py        ← entry point, layout général, routing
    state.py                ← gestion du session_state
    components/sidebar.py   ← upload PDF + paramètres
    components/chat.py      ← chat avec streaming
    components/sources.py   ← panneau des sources avec rendu PDF
"""
from __future__ import annotations

import os

# ──────────────────────────────────────────────────────────────────
# IMPORTANT — Ces variables d'env doivent être set AVANT l'import de torch.
# PyTorch + OpenMP peuvent crasher silencieusement (segfault natif) quand
# le code tourne dans un thread non-principal (cas typique de Streamlit).
# On force PyTorch à 1 thread → pas de problème de threading.
# ──────────────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Empêche un crash particulier sur Windows avec OpenMP en runtime double.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Ajout de src/ au sys.path pour pouvoir importer rag_pdf sans installation editable.
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for p in (str(_SRC), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

from app.components.chat import render_chat
from app.components.sidebar import render_sidebar
from app.components.sources_panel import render_sources_panel
from app.state import StateKey, initialize_state


# ======================================================================
# Configuration de la page (DOIT être la 1re commande Streamlit)
# ======================================================================
st.set_page_config(
    page_title="RAG sur PDF — Formation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Formation RAG — Séance 3. RAG sur PDF avec évaluation et UI professionnelle.",
    },
)


# ======================================================================
# CSS custom — pour une présentation soignée
# ======================================================================
_CSS = """
<style>
    /* Palette cohérente avec le cours */
    :root {
        --primary: #1747aa;
        --secondary: #dc5028;
        --accent: #008c5a;
        --soft: #ebf0f8;
    }

    /* Cache le menu hamburger et le footer "Made with Streamlit" pour un look pro */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Titre principal */
    h1 { color: var(--primary); padding-top: 0; }

    /* Boutons primaires */
    .stButton > button[kind="primary"] {
        background-color: var(--primary);
        border-color: var(--primary);
    }

    /* Messages chat */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Badge "Page X" dans les sources */
    .page-badge {
        display: inline-block;
        background-color: var(--soft);
        color: var(--primary);
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Carte source */
    .source-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        background-color: #fafbfc;
    }

    /* Réduit le padding du conteneur principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Sidebar : un peu plus large par défaut */
    [data-testid="stSidebar"] {
        min-width: 340px;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ======================================================================
# Initialisation du state
# ======================================================================
initialize_state()


# ======================================================================
# Sidebar — toujours visible, contrôle l'ingestion et les paramètres
# ======================================================================
with st.sidebar:
    render_sidebar()


# ======================================================================
# Main — routing selon l'état de la collection
# ======================================================================
st.title("📚 RAG sur PDF")
st.markdown(
    "<p style='color: #666; margin-top: -10px;'>"
    "Pose des questions à ton document. Les réponses citent leurs sources, "
    "et tu peux voir la page exacte du PDF."
    "</p>",
    unsafe_allow_html=True,
)

if not st.session_state[StateKey.COLLECTION_READY]:
    # ────────────────────────────────────────────────────────────
    # Aucun PDF chargé encore — message d'accueil simple.
    # ────────────────────────────────────────────────────────────
    if st.session_state[StateKey.PDF_PATH] is None:
        st.info(
            "👈 **Démarre par charger un PDF** dans la barre latérale.\n\n"
            "Tu peux :\n"
            "- Uploader un PDF (ou utiliser celui placé dans `data/pdfs/`)\n"
            "- Choisir les pages à indexer (ex. `1-10, 15, 20-25`)\n"
            "- Activer la **reformulation de question** et le **reranker**\n"
            "- Régler le `top_k` du retrieval"
        )
        with st.expander("💡 Suggestions de questions pour le PDF du cours", expanded=True):
            st.markdown(
                "- Quelles sont les limites des LLM ?\n"
                "- Qu'est-ce qu'un embedding ?\n"
                "- Quels sont les différents types de RAG ?\n"
                "- Quelle est la différence entre RAG et fine-tuning ?\n"
                "- Quelles métriques utiliser pour évaluer un RAG ?"
            )
        st.stop()

    # ────────────────────────────────────────────────────────────
    # PDF chargé mais pas encore ingéré → on affiche l'APERÇU.
    # ────────────────────────────────────────────────────────────
    from streamlit_pdf_viewer import pdf_viewer

    pdf_path = st.session_state[StateKey.PDF_PATH]
    pdf_name = st.session_state[StateKey.PDF_NAME]
    pdf_pages = st.session_state[StateKey.PDF_PAGE_COUNT]

    col_info, col_preview = st.columns([1, 2], gap="large")

    with col_info:
        st.success(
            f"📄 **{pdf_name}**\n\n"
            f"{pdf_pages} pages détectées."
        )
        st.info(
            "**Étapes suivantes :**\n\n"
            "1. *(Optionnel)* Configurer les **pages à indexer** "
            "dans la sidebar — laisse vide pour tout le PDF.\n"
            "2. *(Optionnel)* Ajuster **chunk size / overlap**.\n"
            "3. Cliquer **🚀 Lancer l'ingestion**.\n\n"
            "L'aperçu à droite te permet de te repérer dans le document avant l'ingestion."
        )
        with st.expander("💡 Suggestions de questions", expanded=False):
            st.markdown(
                "- Quelles sont les limites des LLM ?\n"
                "- Qu'est-ce qu'un embedding ?\n"
                "- Quels sont les différents types de RAG ?\n"
                "- Quelle est la différence entre RAG et fine-tuning ?"
            )

    with col_preview:
        st.markdown("##### 👀 Aperçu du document")
        try:
            pdf_viewer(
                input=str(pdf_path),
                width=700,
                height=820,
                pages_vertical_spacing=4,
                resolution_boost=2,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Impossible d'afficher l'aperçu : {exc}")

    st.stop()

# Une collection est prête → layout chat + sources.
if st.session_state[StateKey.SHOW_SOURCES]:
    col_chat, col_sources = st.columns([3, 2], gap="medium")
    with col_chat:
        render_chat()
    with col_sources:
        render_sources_panel()
else:
    render_chat()
