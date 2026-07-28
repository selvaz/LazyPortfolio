param(
    [int]$Port = 8766,
    [string]$DbPath,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot ".venv-tree-studio\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Tree Studio venv not found at $VenvPython. Run scripts\install_tree_studio.ps1 first."
}

if (-not $DbPath) {
    $DbPath = Resolve-Path (Join-Path $RepoRoot "..\market-data-hub\market_data.duckdb") -ErrorAction Stop
}
if (-not (Test-Path $DbPath -PathType Leaf)) {
    throw "Market Data Hub database not found: $DbPath"
}

$env:MARKET_DATA_DB = [IO.Path]::GetFullPath($DbPath)
Write-Host "Using MARKET_DATA_DB=$($env:MARKET_DATA_DB)" -ForegroundColor DarkGray

$Url = "http://127.0.0.1:$Port/"
Write-Host "LazyPortfolio Tree Studio will run at $Url" -ForegroundColor Green
Write-Host "Stop it with Ctrl+C in this terminal." -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Process $Url
}

Set-Location $RepoRoot
& $VenvPython "project\tree_studio.py" "$Port"
