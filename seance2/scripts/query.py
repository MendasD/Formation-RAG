"""
Script CLI interactif : pose des questions au RAG en boucle (REPL).

Usage :
    uv run python -m scripts.query
"""
import scripts._bootstrap  # noqa: F401  — ajuste le sys.path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from rag_pdf.pipeline.factory import build_rag_chain

console = Console()


def main():
    console.rule("[bold cyan]RAG PDF — Mode interactif")
    console.print(
        "[dim]Tapez vos questions. 'quit' ou Ctrl+C pour quitter.[/dim]\n"
    )

    # Une seule construction de la chaîne — réutilisée pour toutes les questions.
    rag = build_rag_chain()

    while True:
        try:
            question = console.input("[bold yellow]Question :[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question or question.lower() in {"quit", "exit", "q"}:
            break

        try:
            answer = rag.invoke(question)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Erreur :[/] {exc}")
            continue

        console.print(
            Panel(
                Markdown(answer.answer),
                title="[bold green]Réponse",
                border_style="green",
            )
        )
        if answer.unique_pages:
            pages_str = ", ".join(f"Page {p}" for p in answer.unique_pages)
            console.print(f"[dim]Sources : {pages_str}[/]\n")
        else:
            console.print()

    console.print("\n[dim]Au revoir.[/]\n")


if __name__ == "__main__":
    main()
