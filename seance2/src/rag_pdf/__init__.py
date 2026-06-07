"""
rag_pdf — Package du projet RAG sur PDF.

Organisé en sous-modules selon les responsabilités du pipeline RAG :

    ingestion/   → chargement et découpage des documents
    indexing/    → embeddings et vector store
    retrieval/   → recherche des chunks pertinents
    generation/  → LLM et prompts
    pipeline/    → orchestration LCEL et factory
    schemas      → modèles Pydantic partagés
"""

__version__ = "0.1.0"
