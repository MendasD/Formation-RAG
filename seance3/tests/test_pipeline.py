"""
Smoke tests basiques — vérifient que les briques s'instancient correctement.

Ces tests ne valident pas la qualité du RAG (ça, c'est le job de RAGAS
en séance 3). Ils vérifient juste que le code n'est pas cassé : c'est
le filet de sécurité minimum d'un projet pro.

Lancement : uv run pytest
"""
from rag_pdf.generation.prompts import build_rag_prompt
from rag_pdf.ingestion.splitter import Splitter
from rag_pdf.schemas import RAGAnswer, SourceChunk


def test_splitter_basic():
    """Le splitter découpe un texte long en plusieurs chunks."""
    from langchain_core.documents import Document

    doc = Document(
        page_content="Lorem ipsum dolor sit amet. " * 200,
        metadata={"page": 1, "source": "test.pdf"},
    )
    splitter = Splitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split([doc])
    assert len(chunks) > 1
    assert all(c.metadata.get("page") == 1 for c in chunks)


def test_prompt_template_has_required_variables():
    """Le prompt template attend bien `context` et `question`."""
    prompt = build_rag_prompt()
    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables


def test_rag_answer_unique_pages():
    """`unique_pages` déduplique et trie correctement."""
    answer = RAGAnswer(
        question="Q",
        answer="A",
        sources=[
            SourceChunk(page=3, content="..."),
            SourceChunk(page=1, content="..."),
            SourceChunk(page=3, content="..."),
        ],
    )
    assert answer.unique_pages == [1, 3]
