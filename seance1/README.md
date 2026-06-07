# Formation RAG — Séance 1 : Théorie

Ce dossier contient les supports pour la **première séance** de la formation RAG.

## Fichiers

| Fichier | Description |
|---|---|
| `cours_rag_theorie.tex` | Document de cours détaillé (~30-50 pages) à partager aux apprenants |
| `slides_rag_theorie.tex` | Présentation Beamer (thème Metropolis) pour animer la séance |
| `guide_formateur.tex` | Guide du formateur : commentaire pédagogique slide par slide, durées indicatives, astuces d'animation, questions difficiles anticipées |

## Compilation

### Prérequis

Une distribution LaTeX complète :
- **Windows** : MiKTeX ou TeX Live
- **macOS** : MacTeX
- **Linux** : TeX Live (paquet `texlive-full` recommandé)

Le thème Beamer **Metropolis** doit être installé (généralement présent dans les distributions complètes ; sinon : `tlmgr install beamertheme-metropolis`).

### Commandes

Depuis ce dossier, en PowerShell :

```powershell
# Document de cours (2 passes pour la table des matières)
pdflatex cours_rag_theorie.tex
pdflatex cours_rag_theorie.tex

# Présentation Beamer (2 passes également)
pdflatex slides_rag_theorie.tex
pdflatex slides_rag_theorie.tex

# Guide du formateur (2 passes pour la TOC)
pdflatex guide_formateur.tex
pdflatex guide_formateur.tex
```

Alternative avec `latexmk` (gère les passes automatiquement) :

```powershell
latexmk -pdf cours_rag_theorie.tex
latexmk -pdf slides_rag_theorie.tex
```

## Images à récupérer

Les supports contiennent **des emplacements d'images à compléter** (marqués `[IMAGE — ...]` dans le LaTeX). Voici la liste avec la requête Google à utiliser :

| # | Emplacement | Requête Google Images suggérée |
|---|---|---|
| 1 | Diagramme inclusion IA / ML / DL / GenAI / LLM | `AI ML DL GenAI venn diagram` |
| 2 | Visualisation 2D d'embeddings (roi/reine/banane) | `word embeddings visualization` |
| 3 | Architecture générale d'un RAG | `RAG architecture diagram` |
| 4 | Pipeline d'indexation RAG | `RAG indexing pipeline` |
| 5 | Pipeline de requête (retrieval + génération) | `RAG query pipeline` |
| 6 | Espace vectoriel pour la recherche sémantique | `semantic search vector space` |
| 7 | Boucle d'amélioration continue | `evaluation feedback loop` |
| 8 | Illustration de cas d'usage en entreprise | `enterprise AI use cases illustration` |

### Comment insérer une image après l'avoir téléchargée

1. Place l'image dans un sous-dossier `images/` à côté du `.tex`.
2. Remplace le placeholder `\textit{[IMAGE — ...]}` par :

```latex
\includegraphics[width=0.7\textwidth]{images/nom_du_fichier.png}
```

Le package `graphicx` est déjà chargé dans les deux documents.

## Personnalisation rapide

- **Couleurs** : modifier les `\definecolor{...}` en début de chaque fichier (`primary`, `secondary`, `accent`).
- **Auteur / date / titre** : sections `\title`, `\author`, `\date`.
- **Thème Beamer** : remplacer `metropolis` par un autre thème (`Madrid`, `CambridgeUS`, etc.) — attention, les couleurs custom peuvent nécessiter des ajustements.

## Plan de la formation complète

- **Séance 1 — Théorie** (ce dossier) : fondements, fonctionnement, types, évaluation
- **Séance 2 — Pratique 1/2** : construction d'un RAG simple avec LangChain
- **Séance 3 — Pratique 2/2** : évaluation, optimisations, mise en production

---

**Auteur** : Christian Nzonde — Mai 2026
