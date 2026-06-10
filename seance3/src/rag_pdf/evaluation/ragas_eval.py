"""
Évaluation d'un système RAG avec RAGAS.

RAGAS calcule 4 métriques clés :

| Métrique             | Mesure                                                       |
|----------------------|--------------------------------------------------------------|
| `faithfulness`       | La réponse est-elle fidèle au contexte (pas d'hallucination) |
| `answer_relevancy`   | La réponse répond-elle bien à la question                    |
| `context_precision`  | Les chunks récupérés sont-ils pertinents                     |
| `context_recall`     | A-t-on récupéré tous les chunks pertinents                   |

RAGAS utilise lui-même un LLM "juge" pour évaluer ces métriques.

⚠️ Notes pour Groq :
- Groq n'accepte que `n=1` dans son API → certaines métriques RAGAS qui
  demandent `n=3` (answer_relevancy notamment) auront des erreurs partielles.
- Le quota gratuit Groq est de 100k tokens/jour ; RAGAS est gourmand. Sur
  un dataset de 15 questions × 4 configs, vous pouvez l'atteindre.
- Possibilité d'utiliser un modèle plus léger comme juge (par défaut on
  utilise `llama-3.1-8b-instant`, 4× plus de quota que la version 70B).
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# ──────────────────────────────────────────────────────────────────
# Patch important : Groq et Cerebras ne supportent pas `n > 1` dans
# leur API (paramètre OpenAI-compatible "number of completions").
# Par défaut, RAGAS `answer_relevancy` demande 3 reformulations en
# UN seul appel (n=3) — refusé par ces deux providers.
# On force `strictness=1` : on ne génère qu'une seule reformulation
# par question, ce qui rend la métrique légèrement moins précise
# mais compatible avec tous les providers.
# ──────────────────────────────────────────────────────────────────
for _attr in ("strictness", "n"):
    if hasattr(answer_relevancy, _attr):
        try:
            setattr(answer_relevancy, _attr, 1)
        except Exception:  # noqa: BLE001
            pass
        break

from rag_pdf.config import settings
from rag_pdf.generation.llm import make_llm
from rag_pdf.pipeline.factory import CONFIG_NAMES, build_config
from rag_pdf.schemas import EvalResult, GoldenDataset
from rag_pdf.utils.logging import get_logger

if TYPE_CHECKING:
    from rag_pdf.pipeline.rag_chain import RAGChain

logger = get_logger(__name__)


# Modèle utilisé par défaut comme juge RAGAS selon le provider.
# - Sur Groq, on prend un modèle léger (8B) pour ménager le quota free (100k tok/jour).
# - Sur Cerebras free tier, on prend Llama 4 Scout 17B (dispo gratuitement,
#   plus performant que 8B). Pour le 70B il faut un compte Dev payant.
DEFAULT_JUDGE_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "cerebras": "gpt-oss-120b",
}


def _safe_metric(result: Any, key: str) -> float:
    """Extrait une métrique RAGAS de manière robuste.

    RAGAS peut renvoyer :
    - un float (cas nominal)
    - NaN si toutes les questions ont échoué pour cette métrique
    - une liste de scores par question (parfois avec des NaN)
    - une exception (timeout, rate limit…)

    On gère tous ces cas en retournant la moyenne ignorant les NaN, ou NaN
    si vraiment rien d'utilisable.
    """
    try:
        val = result[key] if hasattr(result, "__getitem__") else getattr(result, key, math.nan)
    except (KeyError, AttributeError, TypeError):
        return math.nan

    if val is None:
        return math.nan

    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array([v for v in val if v is not None], dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return math.nan
        return float(np.nanmean(arr))

    try:
        return float(val)
    except (TypeError, ValueError):
        return math.nan


def _run_predictions(rag: "RAGChain", dataset: GoldenDataset) -> dict[str, list]:
    """Exécute le RAG sur toutes les questions du dataset et collecte les réponses + contextes."""
    questions, answers, contexts, gts = [], [], [], []
    latencies = []

    for q in dataset.questions:
        logger.info(f"[dim]Q {q.id} :[/] {q.question[:60]}…")
        t0 = time.perf_counter()
        result = rag.invoke(q.question)
        latencies.append((time.perf_counter() - t0) * 1000)

        questions.append(q.question)
        answers.append(result.answer)
        contexts.append([src.content for src in result.sources])
        gts.append(q.ground_truth)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": gts,
        "_latencies": latencies,
    }


def evaluate_rag(
    rag: "RAGChain",
    dataset: GoldenDataset,
    config_name: str = "custom",
    judge_model: str | None = None,
    judge_provider: str | None = None,
) -> EvalResult:
    """Évalue un RAG sur un golden dataset et retourne les 4 métriques RAGAS + latence.

    Args:
        rag: Le pipeline RAG à évaluer.
        dataset: Le golden dataset.
        config_name: Nom à donner à cette configuration dans le résultat.
        judge_model: Modèle utilisé comme juge RAGAS. Si None, on prend un
            défaut adapté au provider (8B sur Groq pour ménager le quota,
            70B sur Cerebras qui a 1M tokens/jour).
        judge_provider: "groq" ou "cerebras". Si None, lit `settings.llm_provider`.
    """
    logger.info(f"[bold cyan]Évaluation RAGAS — config :[/] {config_name}")

    # 1. Génération des prédictions.
    data = _run_predictions(rag, dataset)
    latencies = data.pop("_latencies")
    mean_latency = sum(latencies) / len(latencies) if latencies else None

    # 2. RAGAS attend un dataset HuggingFace.
    hf_ds = Dataset.from_dict(data)

    # 3. Juge LLM + embeddings (par défaut on n'utilise PAS notre LLM principal
    # comme juge, pour économiser le quota).
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    provider = judge_provider or settings.llm_provider
    judge_name = judge_model or DEFAULT_JUDGE_MODELS.get(provider, "llama-3.1-8b-instant")
    logger.info(f"[dim]Juge RAGAS : {judge_name} (provider : {provider})[/]")

    ragas_llm = LangchainLLMWrapper(
        make_llm(provider=provider, model=judge_name).langchain_llm
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # Config plus prudente : moins de parallélisme, plus de timeout.
    # Évite les bursts qui font sauter le rate limit Groq.
    run_config = RunConfig(
        max_workers=2,        # par défaut 16 → trop pour Groq free tier
        timeout=180,          # 180 s (au lieu de 60 par défaut)
        max_retries=3,
        max_wait=30,
    )

    # 4. Évaluation.
    result = evaluate(
        hf_ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_emb,
        raise_exceptions=False,
        run_config=run_config,
    )

    # 5. Conversion en EvalResult typé — robuste aux échecs partiels.
    eval_result = EvalResult(
        config_name=config_name,
        n_questions=len(dataset),
        faithfulness=_safe_metric(result, "faithfulness"),
        answer_relevancy=_safe_metric(result, "answer_relevancy"),
        context_precision=_safe_metric(result, "context_precision"),
        context_recall=_safe_metric(result, "context_recall"),
        mean_latency_ms=mean_latency,
    )

    # Avertissement si certaines métriques sont NaN
    nan_metrics = [
        m for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        if math.isnan(getattr(eval_result, m))
    ]
    if nan_metrics:
        logger.warning(
            f"[yellow]⚠️  Métriques en NaN (échecs partiels RAGAS) :[/] {nan_metrics}\n"
            f"[dim]Souvent dû à un rate limit Groq ou un timeout. Réessayer plus tard "
            f"ou réduire le dataset.[/]"
        )

    return eval_result


def evaluate_all_configs(
    dataset: GoldenDataset,
    configs: list[str] | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
) -> pd.DataFrame:
    """Évalue les 4 configurations standardisées et retourne un DataFrame comparatif."""
    configs = configs or CONFIG_NAMES
    rows: list[dict] = []

    for cfg_name in configs:
        logger.info(f"\n[bold]═══ Config {cfg_name} ═══[/]")
        rag = build_config(cfg_name)
        result = evaluate_rag(
            rag, dataset,
            config_name=cfg_name,
            judge_model=judge_model,
            judge_provider=judge_provider,
        )
        rows.append(result.model_dump())

    df = pd.DataFrame(rows)
    return df.set_index("config_name")
