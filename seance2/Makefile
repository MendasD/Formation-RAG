# Raccourcis projet — Formation RAG séance 2
.PHONY: help install ingest query notebook clean

help:
	@echo "Cibles disponibles :"
	@echo "  install   - Installe les dépendances avec uv"
	@echo "  ingest    - Indexe le PDF du cours (data/pdfs/cours_rag_theorie.pdf)"
	@echo "  query     - Lance le RAG en mode interactif (CLI)"
	@echo "  notebook  - Ouvre le notebook d'exploration"
	@echo "  clean     - Supprime le cache et la base vectorielle"

install:
	uv sync --extra notebook

ingest:
	uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf

query:
	uv run python -m scripts.query

notebook:
	uv run jupyter notebook notebooks/01_exploration.ipynb

clean:
	rm -rf chroma_db/ __pycache__/ .ipynb_checkpoints/ src/rag_pdf/**/__pycache__/
