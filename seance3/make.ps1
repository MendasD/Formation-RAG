# ============================================================
# make.ps1 — Équivalent PowerShell du Makefile pour Windows
# ============================================================
#
# Usage :
#   .\make.ps1              → affiche l'aide
#   .\make.ps1 install      → installe les dépendances
#   .\make.ps1 ingest       → indexe le PDF du cours
#   .\make.ps1 query        → CLI interactive (REPL)
#   .\make.ps1 app          → lance l'application web Streamlit
#   .\make.ps1 eval         → évaluation RAGAS des 4 configs
#   .\make.ps1 notebook     → ouvre le notebook d'évaluation
#   .\make.ps1 clean        → supprime cache + base vectorielle
#
# Si tu préfères `make` natif, installe-le via :
#   winget install GnuWin32.Make
# (puis redémarre le terminal).

param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "install", "ingest", "query", "app", "eval", "notebook", "clean", IgnoreCase = $true)]
    [string]$Target = "help"
)

# Couleurs pour la sortie
function Write-Step($message) {
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Write-Done($message) {
    Write-Host "[OK] $message" -ForegroundColor Green
}

# ---------- Cibles ----------

function Invoke-Help {
    Write-Host ""
    Write-Host "Cibles disponibles :" -ForegroundColor Yellow
    Write-Host "  .\make.ps1 install     - Installe les dependances avec uv"
    Write-Host "  .\make.ps1 ingest      - Indexe le PDF du cours (toutes les pages)"
    Write-Host "  .\make.ps1 query       - Lance le RAG en mode interactif (CLI)"
    Write-Host "  .\make.ps1 app         - Lance l'application web Streamlit"
    Write-Host "  .\make.ps1 eval        - Lance l'evaluation RAGAS des 4 configs"
    Write-Host "  .\make.ps1 notebook    - Ouvre le notebook d'evaluation"
    Write-Host "  .\make.ps1 clean       - Supprime cache et base vectorielle"
    Write-Host ""
}

function Invoke-Install {
    Write-Step "Installation des dependances (uv sync --extra notebook)"
    uv sync --extra notebook
    if ($LASTEXITCODE -eq 0) { Write-Done "Dependances installees" }
}

function Invoke-Ingest {
    Write-Step "Ingestion du PDF du cours"
    uv run python -m scripts.ingest data/pdfs/cours_rag_theorie.pdf
}

function Invoke-Query {
    Write-Step "Lancement du REPL"
    uv run python -m scripts.query
}

function Invoke-App {
    Write-Step "Lancement de l'app Streamlit (http://localhost:8501)"
    uv run streamlit run app/streamlit_app.py
}

function Invoke-Eval {
    Write-Step "Evaluation RAGAS des 4 configs"
    uv run python -m scripts.evaluate
}

function Invoke-Notebook {
    Write-Step "Ouverture du notebook d'evaluation"
    uv run jupyter notebook notebooks/02_evaluation.ipynb
}

function Invoke-Clean {
    Write-Step "Nettoyage des artefacts"
    $paths = @(
        "chroma_db",
        "__pycache__",
        ".ipynb_checkpoints"
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Write-Host "  - supprime : $p"
        }
    }
    # Caches Python dans tous les sous-dossiers
    Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            Remove-Item -Recurse -Force $_.FullName
            Write-Host "  - supprime : $($_.FullName)"
        }
    # Résultats d'éval CSV
    if (Test-Path "evaluation/results") {
        Get-ChildItem -Path "evaluation/results" -Filter "*.csv" -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
    Write-Done "Nettoyage termine"
}

# ---------- Dispatch ----------

switch ($Target.ToLower()) {
    "help"     { Invoke-Help }
    "install"  { Invoke-Install }
    "ingest"   { Invoke-Ingest }
    "query"    { Invoke-Query }
    "app"      { Invoke-App }
    "eval"     { Invoke-Eval }
    "notebook" { Invoke-Notebook }
    "clean"    { Invoke-Clean }
}
