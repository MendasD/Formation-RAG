"""
Schémas Pydantic partagés dans tout le package.

L'idée : remplacer les `dict` opaques par des objets typés. Cela rend
le code beaucoup plus lisible et permet à votre IDE de vous aider
(autocomplétion, détection d'erreurs).
"""
from typing import Optional

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """Un chunk source associé à une réponse RAG.

    Attributes:
        page: Numéro de page (1-indexé) d'où provient le chunk.
        content: Le texte brut du chunk.
        score: Score de similarité éventuel (1.0 = identique, 0.0 = orthogonal).
        rerank_score: Score post-reranking éventuel (seulement présent si reranker actif).
    """

    page: int = Field(..., description="Numéro de page (1-indexé).")
    content: str = Field(..., description="Contenu textuel du chunk.")
    score: Optional[float] = Field(default=None, description="Score de similarité (cosinus).")
    rerank_score: Optional[float] = Field(default=None, description="Score du reranker, si activé.")


class RAGAnswer(BaseModel):
    """Réponse complète d'un système RAG.

    Contient à la fois le texte généré par le LLM et les chunks
    sources utilisés — c'est ce qui permet la traçabilité.
    """

    question: str = Field(..., description="La question originale de l'utilisateur.")
    rewritten_question: Optional[str] = Field(
        default=None,
        description="Question reformulée par le query_rewriter (si activé).",
    )
    answer: str = Field(..., description="La réponse générée par le LLM.")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Les chunks sources utilisés pour générer la réponse.",
    )
    latency_ms: Optional[float] = Field(default=None, description="Latence totale (ms).")

    @property
    def unique_pages(self) -> list[int]:
        """Pages uniques citées comme sources, triées."""
        return sorted({src.page for src in self.sources})


# ----------------------------------------------------------------------
# Évaluation (séance 3)
# ----------------------------------------------------------------------


class GoldenQuestion(BaseModel):
    """Une question de référence dans le golden dataset, avec sa vérité-terrain.

    Sert à l'évaluation RAGAS : on compare la réponse générée à `ground_truth`
    et les chunks récupérés à `relevant_pages`.
    """

    id: str = Field(..., description="Identifiant unique (ex: 'q01').")
    question: str = Field(..., description="La question telle qu'un utilisateur la poserait.")
    ground_truth: str = Field(..., description="La réponse attendue (validée par un humain).")
    relevant_pages: list[int] = Field(
        default_factory=list,
        description="Pages PDF qui contiennent l'information attendue (1-indexées).",
    )
    category: Optional[str] = Field(
        default=None,
        description="Catégorie (ex: 'definition', 'comparison', 'list', 'out_of_scope').",
    )


class GoldenDataset(BaseModel):
    """Un jeu de questions de référence."""

    name: str = Field(..., description="Nom du dataset (ex: 'cours_rag_v1').")
    questions: list[GoldenQuestion] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)


class EvalResult(BaseModel):
    """Résultat d'une évaluation RAGAS pour une configuration donnée."""

    config_name: str = Field(..., description="Nom de la config (ex: 'A_naive', 'B_rewrite').")
    n_questions: int = Field(..., description="Nombre de questions évaluées.")
    faithfulness: float = Field(..., description="Fidélité au contexte (0-1, plus haut = mieux).")
    answer_relevancy: float = Field(..., description="Pertinence de la réponse vs question.")
    context_precision: float = Field(..., description="Pertinence des chunks récupérés.")
    context_recall: float = Field(..., description="Couverture des chunks pertinents.")
    mean_latency_ms: Optional[float] = Field(default=None, description="Latence moyenne.")
