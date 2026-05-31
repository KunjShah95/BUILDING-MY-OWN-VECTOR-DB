# Local dev: venv, deps, DB init, uvicorn
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "No .venv found. Create one with: python -m venv .venv"
    Write-Host "Then activate: .\.venv\Scripts\Activate.ps1"
    exit 1
}

if (-not $env:VIRTUAL_ENV) {
    Write-Host "Tip: activate the venv first: .\.venv\Scripts\Activate.ps1"
}

Write-Host "Installing requirements..."
& $VenvPython -m pip install -q -r requirements.txt
Write-Host "Installing SDK (editable)..."
& $VenvPython -m pip install -q -e sdk

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Write-Host "Warning: .env missing. Copy .env.example to .env and set DATABASE_URL."
}

Write-Host "Creating database tables..."
& $VenvPython -c "from config.database import Base, engine; Base.metadata.create_all(bind=engine)"

Write-Host "Starting API on http://0.0.0.0:8000 ..."
& $VenvPython -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
