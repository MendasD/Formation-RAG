# Formation RAG — Séance 3 : RAG avancé + app web pro

Cette séance étend le Naive RAG de la séance 2 avec :

- **Évaluation systématique** avec RAGAS (4 métriques : faithfulness, answer_relevancy, context_precision, context_recall)
- **Reformulation de question** (pré-retrieval, `SimpleRewriter` + `HyDERewriter` bonus)
- **Re-ranking** (post-retrieval, cross-encoder `BAAI/bge-reranker-v2-m3` multilingue)
- **Benchmark comparé** de 4 configurations A/B/C/D
- **App web Streamlit professionnelle** : chat avec streaming, affichage des pages PDF sources, sélection des pages à ingérer, paramètres live

À la fin de la séance, vous avez :

1. Un **notebook d'évaluation** (`02_evaluation.ipynb`) qui chiffre l'apport de chaque technique.
2. Un **package `rag_pdf/` enrichi** avec interfaces abstraites pour query rewriting, reranking, évaluation.
3. Un **CLI `scripts/evaluate.py`** pour lancer le benchmark des 4 configs en une commande.
4. Une **app web Streamlit** prête à montrer (et à utiliser).

---

## 1. Prérequis

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)**
- **Une clé API Groq** (gratuite) — réutilise celle de la séance 2.
- **Le PDF du cours** placé dans `data/pdfs/cours_rag_theorie.pdf`.

> Le projet est une **suite logique de la séance 2**. Tout le code de la séance 2 est inclus ici — vous pouvez démarrer sans avoir touché à `seance2/` au préalable.

---

## 2. Installation

```powershell
cd seance3

# Création du venv + installation des dépendances
uv sync --extra notebook

# Copier le template .env et y mettre votre clé Groq (réutilisable depuis seance2/.env)
cp .env.example .env
# Puis éditer .env

# Ingérer le PDF (à faire une fois)
uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf
```

> 🛈 **Pré-télécharger le reranker** (~2 Go, recommandé avant la séance pour éviter d'attendre en live) :
> ```powershell
> uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
> ```

---

## 3. Utilisation

### 🏎️ Lancement rapide

```powershell
make app        # Lance l'app web Streamlit  → http://localhost:8501
make notebook   # Ouvre le notebook d'évaluation
make eval       # Lance le benchmark des 4 configs en CLI
make query      # Mode REPL en terminal (séance 2)
```

### 🌐 App web Streamlit (livrable principal)

```powershell
uv run streamlit run app/streamlit_app.py
```

L'app permet de :

- 📥 **Uploader un PDF** (ou utiliser ceux placés dans `data/pdfs/`)
- 🎯 **Sélectionner les pages à indexer** avec une notation compacte : `1-5, 10, 15-20`
- ⚙️ **Activer/désactiver à la volée** : reformulation de question, reranker
- 🎚️ **Régler le top-k** du retrieval
- 💬 **Chatter avec le document** : réponses streamées token par token
- 📑 **Voir les pages sources** : panneau de droite avec extraits + rendu PNG de la page exacte
- 📄 **Naviguer dans le PDF** : flèches / saut à une page

### 📊 Notebook d'évaluation

```powershell
uv run jupyter notebook notebooks/02_evaluation.ipynb
```

Le notebook déroule pas-à-pas :
1. Chargement du golden dataset (15 questions annotées).
2. Évaluation RAGAS de la baseline (Naive RAG, config A).
3. Ajout de la reformulation (config B).
4. Ajout du reranker (config C).
5. Cumul des deux (config D).
6. Tableau de synthèse coloré + visualisations comparatives.
7. (Bonus) Démo d'extension à Pinecone en 10 lignes.

### 🧪 Benchmark CLI

```powershell
uv run python -m scripts.evaluate
# → affiche un tableau Rich + sauvegarde un CSV dans evaluation/results/
```

---

## 4. Structure du projet

```
seance3/
├── pyproject.toml + .env.example + Makefile + .gitignore + README.md
├── guide_formateur_seance3.md
│
├── src/rag_pdf/                       ← le package métier
│   ├── config.py                       ← config centralisée (+ flags A/B/C/D)
│   ├── schemas.py                      ← + GoldenQuestion, EvalResult
│   ├── ingestion/
│   │   ├── loader.py                   ← + sélection de pages
│   │   └── splitter.py
│   ├── indexing/
│   │   ├── embedder.py
│   │   └── vector_store.py
│   ├── retrieval/
│   │   ├── retriever.py                ← + intégration reranker + filtre pages
│   │   ├── query_rewriter.py           ← NEW : Simple/HyDE/NoOp
│   │   └── reranker.py                 ← NEW : CrossEncoder/NoOp
│   ├── generation/
│   │   ├── llm.py
│   │   └── prompts.py                  ← + REWRITE & HYDE prompts
│   ├── pipeline/
│   │   ├── rag_chain.py                ← + query_rewriter + latence
│   │   └── factory.py                  ← + flags A/B/C/D, build_config()
│   ├── evaluation/                     ← NEW
│   │   ├── dataset.py                  ← Golden dataset (Pydantic + JSON)
│   │   └── ragas_eval.py               ← Wrapper RAGAS multi-configs
│   └── utils/
│       ├── logging.py
│       ├── page_parser.py              ← NEW : parse "1-5, 10, 15-20"
│       └── pdf_renderer.py             ← NEW : PDF → PNG cached
│
├── app/                                ← NEW — App web Streamlit
│   ├── streamlit_app.py                ← Entry point
│   ├── state.py                        ← Gestion session_state
│   └── components/
│       ├── sidebar.py                  ← Upload + ingestion + paramètres
│       ├── chat.py                     ← Chat avec streaming
│       └── sources_panel.py            ← Sources + rendu PDF
│
├── scripts/
│   ├── ingest.py                       ← + options --pages, --reset
│   ├── query.py
│   └── evaluate.py                     ← NEW : benchmark 4 configs en table
│
├── notebooks/
│   ├── 01_exploration.ipynb            ← reprise séance 2 (référence)
│   └── 02_evaluation.ipynb             ← NEW : éval RAGAS comparée
│
├── evaluation/                         ← NEW
│   ├── golden_dataset.json             ← 15 questions de référence
│   └── results/                        ← CSVs générés
│
├── data/pdfs/                          ← place le PDF ici
├── chroma_db/                          ← base vectorielle persistée
└── tests/
```

---

## 5. Les 4 configurations comparées

```python
from rag_pdf.pipeline.factory import build_config

# A — Naive (baseline, séance 2)
rag_A = build_config("A_naive")

# B — + Reformulation de question (pré-retrieval)
rag_B = build_config("B_rewrite")

# C — + Reranker (post-retrieval)
rag_C = build_config("C_rerank")

# D — Combiné
rag_D = build_config("D_combined")
```

Chaque config se construit avec **les mêmes briques** — c'est le bénéfice des interfaces abstraites posées en séance 2.

---

## 6. Dépannage

- **App Streamlit lente au 1er chargement** : téléchargement du modèle d'embeddings + reranker en arrière-plan. Patience la 1re fois.
- **`AssertionError: collection vide`** : exécuter `make ingest` ou utiliser la sidebar de l'app pour ingérer.
- **RAGAS lent** : 30-60 s par question évaluée, × 15 questions × 4 configs ≈ 30-60 min total. Tester d'abord sur 3-5 questions.
- **Streamlit demande un email** au lancement : ignorez (`Skip`) ou répondez. C'est l'opt-in télémétrie.
- **Erreur `Pinecone` à l'import** : le code Pinecone est en bonus dans le notebook 02, pas installé par défaut. Voir la section bonus du notebook.

---

## 7. Pour aller plus loin

- **LangSmith / Phoenix** : observabilité production (callbacks LangChain → 2 lignes à ajouter).
- **LangGraph** : pour l'agentic RAG (le RAG décide lui-même de re-chercher si insatisfait).
- **DSPy** : optimisation automatique des prompts.
- **Voyage AI** / **Cohere embed-v3** : embeddings premium si on a un budget.
