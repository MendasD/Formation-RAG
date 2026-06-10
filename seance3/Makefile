# Raccourcis projet — Formation RAG séance 3
.PHONY: help install ingest query app eval notebook clean

help:
	@echo "Cibles disponibles :"
	@echo "  install     - Installe les dépendances avec uv"
	@echo "  ingest      - Indexe le PDF du cours (toutes les pages)"
	@echo "  query       - Lance le RAG en mode interactif (CLI)"
	@echo "  app         - Lance l'application web Streamlit"
	@echo "  eval        - Lance l'évaluation RAGAS des 4 configs"
	@echo "  notebook    - Ouvre le notebook d'évaluation"
	@echo "  clean       - Supprime cache et base vectorielle"

install:
	uv sync --extra notebook

ingest:
	uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf

query:
	uv run python -m scripts.query

app:
	uv run streamlit run app/streamlit_app.py

eval:
	uv run python -m scripts.evaluate

notebook:
	uv run jupyter notebook notebooks/02_evaluation.ipynb

clean:
	rm -rf chroma_db/ __pycache__/ .ipynb_checkpoints/ src/rag_pdf/**/__pycache__/ evaluation/results/*.csv
