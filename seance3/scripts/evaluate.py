"""
Script CLI d'évaluation : compare les 4 configurations RAG sur le golden dataset.

Usage :
    uv run python -m scripts.evaluate
    uv run python -m scripts.evaluate --configs A_naive B_rewrite
    uv run python -m scripts.evaluate --dataset evaluation/golden_dataset.json
"""
import scripts._bootstrap  # noqa: F401

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rag_pdf.config import settings
from rag_pdf.evaluation.dataset import load_golden_dataset
from rag_pdf.evaluation.ragas_eval import evaluate_all_configs
from rag_pdf.pipeline.factory import CONFIG_NAMES

app = typer.Typer(add_completion=False, help="Évaluation RAGAS comparée des configs RAG.")
console = Console()


@app.command()
def main(
    dataset_path: Path = typer.Option(
        Path("evaluation/golden_dataset.json"),
        "--dataset",
        "-d",
        help="Chemin vers le golden dataset JSON.",
    ),
    configs: Optional[list[str]] = typer.Option(
        None,
        "--configs",
        "-c",
        help=f"Configs à évaluer (par défaut : les 4). Choix : {CONFIG_NAMES}",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Chemin du CSV de sortie (par défaut : evaluation/results/results.csv).",
    ),
):
    """Lance l'évaluation comparée et sauvegarde les résultats."""
    if not dataset_path.exists():
        console.print(f"[red]✗ Dataset introuvable :[/] {dataset_path}")
        raise typer.Exit(code=1)

    console.rule("[bold cyan]Évaluation RAGAS — 4 configs comparées")
    console.print(f"[dim]Dataset :[/] {dataset_path}")

    dataset = load_golden_dataset(dataset_path)
    console.print(f"[dim]Nombre de questions :[/] {len(dataset)}\n")

    df = evaluate_all_configs(dataset, configs=configs)

    # Affichage en table Rich.
    table = Table(title="Comparaison des configurations RAG", show_lines=True)
    table.add_column("Config", style="bold cyan")
    for col in df.columns:
        table.add_column(col, justify="right")
    for cfg_name, row in df.iterrows():
        table.add_row(
            cfg_name,
            *[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row],
        )
    console.print("\n", table)

    # Sauvegarde CSV.
    out = output or settings.evaluation_dir / "results" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    console.print(f"\n[bold green]✓ Résultats sauvegardés :[/] {out}")


if __name__ == "__main__":
    app()
