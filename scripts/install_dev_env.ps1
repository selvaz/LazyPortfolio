param(
    [string]$Python = "C:\ProgramData\spyder-6\python.exe",
    [switch]$WithLazyFin
)

# ============================================================================
# install_dev_env.ps1 - install LazyPortfolio (and optionally LazyFin) into an
# existing Python environment, no venv.
#
# This targets the same global Python (spyder-6, has duckdb/pandas/numpy/
# scipy/skfolio/pydantic preinstalled) that market-data-hub's own
# setup_first_run.ps1 defaults to, so everything lands in ONE environment and
# `import market_data_hub` works from here too. Run market-data-hub's own
# installer separately (see notes at the end) - this script does not touch it.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_dev_env.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_dev_env.ps1 -WithLazyFin
# ============================================================================

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if (-not (Test-Path $Python)) {
    throw "Python not found at $Python. Pass -Python <path> to point at your interpreter."
}

function Invoke-Step {
    param([string]$Title, [scriptblock]$Command)
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title" }
}

Invoke-Step "Upgrading pip" {
    & $Python -m pip install --upgrade pip
}

Invoke-Step "Installing LazyPortfolio [dev] (pydantic, numpy, pandas, scipy, skfolio, pytest, ruff, mypy)" {
    & $Python -m pip install -e "$RepoRoot[dev]"
}

if ($WithLazyFin) {
    $LazyFinPath = Resolve-Path (Join-Path $RepoRoot "..\LazyFin") -ErrorAction SilentlyContinue
    if (-not $LazyFinPath) {
        throw "LazyFin checkout not found next to LazyPortfolio. Clone it or omit -WithLazyFin."
    }
    Invoke-Step "Installing LazyFin [dev] from $LazyFinPath" {
        & $Python -m pip install -e "$($LazyFinPath.Path)[dev]"
    }
}

Write-Host ""
Write-Host "==> Done." -ForegroundColor Green
Write-Host ""
Write-Host "LazyPortfolio's own test suite needs NO market-data-hub / no live db -" -ForegroundColor Yellow
Write-Host "every test builds its returns in-memory (numpy/pandas). Run it with:"
Write-Host "  $Python -m pytest -q"
Write-Host ""
Write-Host "market-data-hub is only needed to run against REAL data (Tree Studio," -ForegroundColor Yellow
Write-Host "portfolio_optimizer_run/_backtest over live prices). This script does NOT"
Write-Host "install it - run market-data-hub's own installer next:"
Write-Host ""
Write-Host "  cd ..\market-data-hub"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup_first_run.ps1 -Python `"$Python`""
Write-Host ""
Write-Host "Passing the SAME -Python here keeps it in one environment; its default" -ForegroundColor DarkGray
Write-Host "(-Python C:\ProgramData\spyder-6\python.exe) already matches this script's" -ForegroundColor DarkGray
Write-Host "default, so if you didn't override -Python above you can drop it there too." -ForegroundColor DarkGray
Write-Host ""
$ExistingDb = Join-Path (Split-Path $RepoRoot -Parent) "market-data-hub\market_data.duckdb"
if (Test-Path $ExistingDb) {
    Write-Host "An existing populated database was found:" -ForegroundColor Yellow
    Write-Host "  $ExistingDb"
    Write-Host "Point LazyPortfolio at it with:"
    Write-Host "  `$env:MARKET_DATA_DB = `"$ExistingDb`""
    Write-Host "(setup_first_run.ps1 -DbPath defaults to market_data.duckdb in its own repo" -ForegroundColor DarkGray
    Write-Host "root, i.e. this same file - re-running it against the existing db is safe" -ForegroundColor DarkGray
    Write-Host "and idempotent; it will not recreate it from scratch)." -ForegroundColor DarkGray
} else {
    Write-Host "No existing market_data.duckdb found next to market-data-hub." -ForegroundColor Yellow
    Write-Host "setup_first_run.ps1 will create a fresh one (-DbPath market_data.duckdb by default)."
}
