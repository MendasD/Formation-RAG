"""
Petit utilitaire pour permettre aux scripts CLI d'importer le package
`rag_pdf` lorsqu'on les lance via `python -m scripts.xxx`.

Normalement, `uv sync` installe `rag_pdf` en mode editable et cet ajout
de sys.path n'est pas nécessaire. On le garde par sécurité pour les cas
où le package ne serait pas installé (clone fraîchement, debug…).

Tous les scripts CLI importent ce module en premier.
"""
import sys
from pathlib import Path

# Le dossier src/ contenant le package rag_pdf
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
