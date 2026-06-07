# Formation RAG — Séance 2 : construire un RAG fonctionnel

Cette séance construit pas-à-pas un **Naive RAG** sur un PDF, avec :

- **LangChain** pour l'orchestration
- **PyMuPDF** pour charger le PDF (en conservant les numéros de page)
- **HuggingFace** (`intfloat/multilingual-e5-base`) pour les embeddings multilingues
- **ChromaDB** comme vector store local persistant
- **Groq** (Llama 3.3 70B) comme LLM de génération — rapide et gratuit

À la fin de la séance, vous avez :

1. Un **notebook pas-à-pas** qui montre chaque brique du RAG avec son rendu.
2. Un **package Python `rag_pdf/`** avec la structure professionnelle.
3. Un **script CLI** pour ingérer un PDF et interroger le RAG depuis le terminal.

> La séance 3 ajoutera : reformulation de question, re-ranking, évaluation RAGAS, et une app web Streamlit qui affiche les pages PDF sources.

---

## 1. Prérequis

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** comme gestionnaire d'environnement et de paquets.
  Installation : `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows) ou `curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/macOS).
- **Une clé API Groq** : créez un compte gratuit sur <https://console.groq.com> puis générez une clé sur <https://console.groq.com/keys>.
- **Le PDF du cours de la séance 1** : compilez `../cours_rag_theorie.tex` en PDF (deux passes `pdflatex`) puis placez-le dans `data/pdfs/cours_rag_theorie.pdf`.

---

## 2. Installation

```powershell
cd seance2

# Création du venv + installation des dépendances
uv sync --extra notebook

# Copier le template .env et y mettre votre clé Groq
cp .env.example .env
# Puis éditer .env pour mettre GROQ_API_KEY=gsk_...
```

---

## 3. Utilisation

### Option A — Notebook pédagogique (séance pas-à-pas)

```powershell
uv run jupyter notebook notebooks/01_exploration.ipynb
```

Le notebook déroule chaque brique du RAG (loader, splitter, embeddings, vector store, retriever, LLM, chaîne complète), avec le rendu visible à chaque étape. La dernière section montre comment **réutiliser le package refactoré** en 3 lignes de code.

### Option B — Ligne de commande (workflow "vrai projet")

```powershell
# 1. Indexer le PDF (à faire une seule fois)
uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf

# 2. Interroger le RAG en mode REPL
uv run python -m scripts.query
```

Ou via le `Makefile` (si `make` est dispo) : `make install`, `make ingest`, `make query`.

---

## 4. Structure du projet

```
seance2/
├── pyproject.toml              ← dépendances + métadonnées (PEP 621)
├── .env.example                ← template pour les clés API
├── Makefile                    ← raccourcis (install, ingest, query)
│
├── config/
│   └── settings.py             ← config centralisée (pydantic-settings)
│
├── src/rag_pdf/                ← le package importable
│   ├── schemas.py              ← Pydantic : Document, SourceChunk, RAGAnswer
│   ├── ingestion/
│   │   ├── loader.py           ← PDFLoader (wrapper PyMuPDF)
│   │   └── splitter.py         ← stratégie de chunking
│   ├── indexing/
│   │   ├── embedder.py         ← interface + HFEmbedder
│   │   └── vector_store.py     ← interface + ChromaVectorStore
│   ├── retrieval/
│   │   └── retriever.py        ← similarité vectorielle (top-k)
│   ├── generation/
│   │   ├── llm.py              ← interface + GroqLLM
│   │   └── prompts.py          ← templates de prompt versionnés
│   ├── pipeline/
│   │   ├── rag_chain.py        ← assemblage LCEL
│   │   └── factory.py          ← build_rag_chain() depuis la config
│   └── utils/
│       └── logging.py
│
├── scripts/
│   ├── ingest.py               ← CLI d'ingestion
│   └── query.py                ← CLI interactif (REPL)
│
├── notebooks/
│   └── 01_exploration.ipynb    ← le support pédagogique de la séance
│
├── data/pdfs/                  ← corpus (PDF du cours à déposer ici)
├── chroma_db/                  ← base vectorielle persistée (généré)
└── guide_formateur_seance2.md  ← guide d'animation pour le formateur
```

---

## 5. Patterns "pro" mis en œuvre

| Pattern | Où | Pourquoi |
|---|---|---|
| **Interfaces abstraites + impls** | `embedder.py`, `vector_store.py`, `llm.py` | Pour swapper Chroma↔Pinecone ou Groq↔OpenAI sans toucher au reste (séance 3) |
| **Config centralisée** | `config/settings.py` | Une seule source de vérité, validée au démarrage, lue depuis `.env` |
| **Schémas typés** (Pydantic) | `schemas.py` | Plus de `dict` opaques |
| **Prompts versionnés** | `prompts.py` | Les templates sont du code, pas des strings perdues |
| **CLI entry points** | `scripts/` | On lance comme une vraie app |
| **Logging structuré** | `utils/logging.py` | `logging` + `rich` au lieu de `print()` |

---

## 6. Dépannage

- **`GROQ_API_KEY` manquante** : vérifiez votre `.env` (pas `.env.example`).
- **Téléchargement lent du modèle d'embedding** : la première exécution télécharge `multilingual-e5-base` (~1 Go) depuis HuggingFace. Une fois en cache (`~/.cache/huggingface/`), c'est instantané.
- **`ImportError` sur Chroma** : `uv sync` doit avoir installé `langchain-chroma` + `chromadb`. Sinon : `uv add langchain-chroma chromadb`.
- **Le PDF n'est pas trouvé** : il doit être dans `data/pdfs/cours_rag_theorie.pdf`. Compilez le `.tex` de la séance 1 d'abord.
