"""
Gestion du golden dataset — questions de référence pour l'évaluation.

Un golden dataset est constitué de questions accompagnées de :
- la réponse attendue (`ground_truth`), validée par un expert métier ;
- éventuellement, les pages PDF où l'information se trouve (`relevant_pages`).

Format de stockage : JSON, validé par Pydantic. Simple, versionnable, éditable.
"""
import json
from pathlib import Path

from rag_pdf.schemas import GoldenDataset, GoldenQuestion


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """Charge un golden dataset depuis un fichier JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return GoldenDataset(**data)


def save_golden_dataset(dataset: GoldenDataset, path: str | Path) -> None:
    """Sauvegarde un golden dataset au format JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataset.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_question(
    qid: str,
    question: str,
    ground_truth: str,
    relevant_pages: list[int] | None = None,
    category: str | None = None,
) -> GoldenQuestion:
    """Helper pour construire rapidement une GoldenQuestion."""
    return GoldenQuestion(
        id=qid,
        question=question,
        ground_truth=ground_truth,
        relevant_pages=relevant_pages or [],
        category=category,
    )
