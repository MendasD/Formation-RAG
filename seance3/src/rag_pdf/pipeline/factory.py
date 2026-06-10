"""
Factory : construit le pipeline RAG complet à partir de la config.

C'est ici qu'on assemble tous les composants. Un seul `build_rag_chain()`
suffit pour récupérer un pipeline fonctionnel : c'est l'API publique
qu'utilisera tout le reste du projet (scripts, notebooks, app Streamlit).

Évolution séance 3 :
- Support des configs A/B/C/D via les flags `use_query_rewriting` et `use_reranker`.
- Tous les paramètres peuvent être surchargés (utile pour les benchmarks dans le notebook).

Configurations standardisées (cf notebook 02_evaluation) :
    A — Naive    : use_query_rewriting=False, use_reranker=False
    B — +Rewrite : use_query_rewriting=True,  use_reranker=False
    C — +Rerank  : use_query_rewriting=False, use_reranker=True
    D — Combiné  : use_query_rewriting=True,  use_reranker=True
"""
from typing import Optional

from rag_pdf.config import settings
from rag_pdf.generation.llm import LLMProvider, make_llm
from rag_pdf.indexing.embedder import HFEmbedder
from rag_pdf.indexing.vector_store import ChromaVectorStore
from rag_pdf.pipeline.rag_chain import RAGChain
from rag_pdf.retrieval.query_rewriter import (
    BaseQueryRewriter,
    HyDERewriter,
    NoOpRewriter,
    SimpleRewriter,
)
from rag_pdf.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    NoOpReranker,
)
from rag_pdf.retrieval.retriever import Retriever


def build_rag_chain(
    *,
    use_query_rewriting: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
    rewriter_strategy: str = "simple",
    top_k: Optional[int] = None,
    top_k_initial: Optional[int] = None,
    collection_name: Optional[str] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> RAGChain:
    """Construit un pipeline RAG complet depuis la configuration centralisée.

    Args:
        use_query_rewriting: Active la reformulation de question (pré-retrieval).
            Si None, lit `settings.use_query_rewriting`.
        use_reranker: Active le reranker (post-retrieval).
            Si None, lit `settings.use_reranker`.
        rewriter_strategy: "simple" (par défaut) ou "hyde".
        top_k: Override du nombre de chunks finaux.
        top_k_initial: Override du nombre de chunks remontés avant reranking.
        collection_name: Override du nom de la collection Chroma
            (utile pour tester sur des configs/datasets différents).
        llm_provider: "groq" ou "cerebras". Si None, lit `settings.llm_provider`.

    Returns:
        Une instance de `RAGChain` prête à être interrogée via `.invoke()` ou `.stream()`.

    Préconditions :
        La collection Chroma doit avoir été remplie via le script d'ingestion :
            `python -m scripts.ingest data/pdfs/<votre_pdf>.pdf`
    """
    # Résolution des paramètres : argument explicite > settings par défaut.
    use_query_rewriting = (
        use_query_rewriting if use_query_rewriting is not None else settings.use_query_rewriting
    )
    use_reranker = use_reranker if use_reranker is not None else settings.use_reranker
    top_k = top_k if top_k is not None else settings.top_k
    top_k_initial = top_k_initial if top_k_initial is not None else settings.top_k_initial
    collection_name = collection_name or settings.collection_name

    # 1. Briques de base.
    embedder = HFEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    vector_store = ChromaVectorStore(
        embedder=embedder,
        persist_directory=settings.chroma_persist_dir,
        collection_name=collection_name,
    )
    # LLM via la factory multi-provider (Groq ou Cerebras selon settings/argument).
    llm = make_llm(provider=llm_provider)

    # 2. Reranker (optionnel).
    reranker: BaseReranker
    if use_reranker:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=settings.embedding_device,
        )
    else:
        reranker = NoOpReranker()

    # 3. Retriever (qui intègre le reranker).
    retriever = Retriever(
        vector_store=vector_store,
        k=top_k,
        k_initial=top_k_initial,
        reranker=reranker,
    )

    # 4. Query rewriter (optionnel).
    rewriter: BaseQueryRewriter
    if use_query_rewriting:
        rewriter_class = {"simple": SimpleRewriter, "hyde": HyDERewriter}[rewriter_strategy]
        rewriter = rewriter_class(llm=llm.langchain_llm)
    else:
        rewriter = NoOpRewriter()

    # 5. Assemblage final.
    return RAGChain(retriever=retriever, llm=llm, query_rewriter=rewriter)


# ----------------------------------------------------------------------
# Helpers pour les 4 configurations standardisées (benchmark séance 3)
# ----------------------------------------------------------------------

CONFIG_NAMES = ["A_naive", "B_rewrite", "C_rerank", "D_combined"]


def build_config(name: str) -> RAGChain:
    """Construit l'une des 4 configurations standardisées du benchmark.

    Args:
        name: "A_naive", "B_rewrite", "C_rerank" ou "D_combined".
    """
    configs = {
        "A_naive":    {"use_query_rewriting": False, "use_reranker": False},
        "B_rewrite":  {"use_query_rewriting": True,  "use_reranker": False},
        "C_rerank":   {"use_query_rewriting": False, "use_reranker": True},
        "D_combined": {"use_query_rewriting": True,  "use_reranker": True},
    }
    if name not in configs:
        raise ValueError(f"Config inconnue : {name}. Attendu : {list(configs)}")
    return build_rag_chain(**configs[name])
