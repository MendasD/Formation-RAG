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
    """

    page: int = Field(..., description="Numéro de page (1-indexé).")
    content: str = Field(..., description="Contenu textuel du chunk.")
    score: Optional[float] = Field(default=None, description="Score de similarité.")


class RAGAnswer(BaseModel):
    """Réponse complète d'un système RAG.

    Contient à la fois le texte généré par le LLM et les chunks
    sources utilisés — c'est ce qui permet la traçabilité.
    """

    question: str = Field(..., description="La question originale de l'utilisateur.")
    answer: str = Field(..., description="La réponse générée par le LLM.")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Les chunks sources utilisés pour générer la réponse.",
    )

    @property
    def unique_pages(self) -> list[int]:
        """Pages uniques citées comme sources, triées."""
        return sorted({src.page for src in self.sources})
