"""
Script CLI d'ingestion : charge un PDF dans le vector store Chroma.

Usage :
    uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf
    uv run python -m scripts.ingest data/pdfs/cours.pdf --pages "1-5, 10, 15-20"
    uv run python -m scripts.ingest data/pdfs/cours.pdf --reset
"""
import scripts._bootstrap  # noqa: F401  — ajuste le sys.path

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rag_pdf.config import settings
from rag_pdf.indexing.embedder import HFEmbedder
from rag_pdf.indexing.vector_store import ChromaVectorStore
from rag_pdf.ingestion.loader import PDFLoader
from rag_pdf.ingestion.splitter import Splitter
from rag_pdf.utils.page_parser import PageRangeError, parse_page_ranges

app = typer.Typer(add_completion=False, help="Ingestion d'un PDF dans le vector store.")
console = Console()


@app.command()
def main(
    pdf_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Chemin vers le PDF à indexer.",
    ),
    pages: Optional[str] = typer.Option(
        None,
        "--pages",
        "-p",
        help='Pages à indexer, ex: "1-5, 10, 15-20". Vide = tout le PDF.',
    ),
    chunk_size: int = typer.Option(
        settings.chunk_size, help="Taille cible d'un chunk (caractères)."
    ),
    chunk_overlap: int = typer.Option(
        settings.chunk_overlap, help="Chevauchement entre chunks."
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Vider la collection avant ingestion (réindexation complète).",
    ),
):
    """Charge un PDF, le découpe, l'embedde, et le persiste dans Chroma."""
    console.rule(f"[bold cyan]Ingestion : {pdf_path.name}")

    # 0. Reset éventuel.
    if reset and settings.chroma_persist_dir.exists():
        shutil.rmtree(settings.chroma_persist_dir)
        console.print(f"[yellow]✓ Collection précédente supprimée[/]")

    # 1. Parsing des pages (si spécifié).
    page_list = None
    if pages:
        try:
            page_list = parse_page_ranges(pages)
            console.print(f"[cyan]Pages demandées :[/] {page_list}")
        except PageRangeError as exc:
            console.print(f"[red]✗ Erreur de plages de pages :[/] {exc}")
            raise typer.Exit(code=1)

    # 2. Chargement.
    loader = PDFLoader()
    documents = loader.load(pdf_path, pages=page_list)

    # 3. Chunking.
    splitter = Splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split(documents)

    # 4. Embeddings + indexation.
    embedder = HFEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    vector_store = ChromaVectorStore(
        embedder=embedder,
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )
    vector_store.add_documents(chunks)

    console.print(
        f"\n[bold green]✓ Base persistée[/] dans : "
        f"[yellow]{settings.chroma_persist_dir}[/]"
    )
    console.rule("[bold green]Ingestion terminée")


if __name__ == "__main__":
    app()
