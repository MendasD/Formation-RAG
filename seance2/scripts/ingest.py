"""
Script CLI d'ingestion : charge un PDF dans le vector store Chroma.

Usage :
    uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf
"""
import scripts._bootstrap  # noqa: F401  — ajuste le sys.path

from pathlib import Path
import typer
from rich.console import Console

from rag_pdf.config import settings
from rag_pdf.indexing.embedder import HFEmbedder
from rag_pdf.indexing.vector_store import ChromaVectorStore
from rag_pdf.ingestion.loader import PDFLoader
from rag_pdf.ingestion.splitter import Splitter

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
    chunk_size: int = typer.Option(
        settings.chunk_size, help="Taille cible d'un chunk (caractères)."
    ),
    chunk_overlap: int = typer.Option(
        settings.chunk_overlap, help="Chevauchement entre chunks."
    ),
):
    """Charge un PDF, le découpe, l'embedde, et le persiste dans Chroma."""
    console.rule(f"[bold cyan]Ingestion : {pdf_path.name}")

    # 1. Chargement
    loader = PDFLoader()
    documents = loader.load(pdf_path)

    # 2. Chunking
    splitter = Splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split(documents)

    # 3. Embeddings + indexation
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
