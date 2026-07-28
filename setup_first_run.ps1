# ============================================================================
# setup_first_run.ps1 - guided first-run setup for LazyPortfolio
#
# Run from PowerShell, from the repo root:
#   powershell -ExecutionPolicy Bypass -File .\setup_first_run.ps1
#
# Installs the package, always wires up market-data-hub (the preferred way
# every function in this repo consumes price/return data - never optional in
# spirit, even though it's an installable "extra"), asks only about the
# genuinely optional pieces (dev/test tooling, Tree Studio's JS test harness),
# and locates/records the Market Data Hub database this environment should
# use. Idempotent - safe to re-run after pulling an update or on a new machine.
# ============================================================================
param(
    [string]$Python = "python",
    [string]$MarketDataHubPath,
    [string]$DbPath,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot ".")

function Invoke-Step {
    param([string]$Title, [scriptblock]$Command)
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title" }
}

function Read-YesNo {
    param([string]$Prompt, [bool]$DefaultYes = $false)
    $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
    $answer = Read-Host "$Prompt $suffix"
    if (-not $answer) { return $DefaultYes }
    return $answer -match '^(y|yes|s|si|sì)$'
}

# ---------------------------------------------------------------------------
# 1. Verify Python
# ---------------------------------------------------------------------------
Invoke-Step "Checking Python" {
    & $Python --version
}

Invoke-Step "Upgrading pip" {
    & $Python -m pip install --upgrade pip
}

# ---------------------------------------------------------------------------
# 2. Core install (pydantic, numpy, pandas, scipy, skfolio - always required,
#    declared in pyproject.toml, no prompt needed)
# ---------------------------------------------------------------------------
Invoke-Step "Installing LazyPortfolio" {
    & $Python -m pip install -e "$RepoRoot"
}

# ---------------------------------------------------------------------------
# 3. market-data-hub: ALWAYS installed, never asked. It is the one supported
#    way every LazyPortfolio function reads price/return data - treat it as
#    core, not an optional add-on, regardless of how it's packaged.
# ---------------------------------------------------------------------------
$SiblingHub = if ($MarketDataHubPath) {
    Resolve-Path $MarketDataHubPath -ErrorAction Stop
} else {
    Resolve-Path (Join-Path $RepoRoot "..\market-data-hub") -ErrorAction SilentlyContinue
}

if ($SiblingHub) {
    Invoke-Step "Installing local market-data-hub checkout ($($SiblingHub.Path))" {
        & $Python -m pip install -e $SiblingHub.Path
    }
} else {
    Invoke-Step "Installing market-data-hub from GitHub" {
        & $Python -m pip install "market-data-hub @ git+https://github.com/selvaz/market-data-hub.git"
    }
    Write-Host ""
    Write-Host "No local market-data-hub checkout found next to LazyPortfolio." -ForegroundColor Yellow
    Write-Host "That's fine for using the package, but you'll need a populated .duckdb" -ForegroundColor Yellow
    Write-Host "database (see -DbPath below) and won't get market-data-hub's own" -ForegroundColor Yellow
    Write-Host "ingestion/scheduling scripts. Clone https://github.com/selvaz/market-data-hub" -ForegroundColor Yellow
    Write-Host "next to this repo and run ITS setup_first_run.ps1 for the full data pipeline." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 4. Optional extras - genuinely optional, so ask.
# ---------------------------------------------------------------------------
if (Read-YesNo "Install dev/test tooling (pytest, ruff, mypy)?" $true) {
    Invoke-Step "Installing dev tooling" {
        & $Python -m pip install -e "$RepoRoot[dev]"
    }
    $HaveDev = $true
} else {
    $HaveDev = $false
}

$HaveNode = $false
if (Get-Command npm -ErrorAction SilentlyContinue) {
    if (Read-YesNo "Install Tree Studio's JS test harness (npm install)?" $false) {
        Invoke-Step "Running npm install" {
            Push-Location $RepoRoot
            try { npm install } finally { Pop-Location }
        }
        $HaveNode = $true
    }
} else {
    Write-Host ""
    Write-Host "npm not found - skipping Tree Studio's JS test harness (optional, only" -ForegroundColor DarkGray
    Write-Host "needed to run 'npm test' against project/tree_studio.html)." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 5. Locate / configure the Market Data Hub database (MARKET_DATA_DB)
# ---------------------------------------------------------------------------
function Select-MarketDataDb {
    param([string]$RequestedPath, $SiblingHub)

    if ($RequestedPath) {
        return [IO.Path]::GetFullPath($RequestedPath)
    }
    $existing = [Environment]::GetEnvironmentVariable("MARKET_DATA_DB", "User")
    if ($existing -and (Test-Path $existing -PathType Leaf)) {
        return $existing
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($SiblingHub) {
        Get-ChildItem -LiteralPath $SiblingHub.Path -Filter "*.duckdb" -File -ErrorAction SilentlyContinue |
            ForEach-Object { [void]$candidates.Add($_.FullName) }
    }
    $candidates = @($candidates | Select-Object -Unique)

    Write-Host ""
    Write-Host "Market Data Hub database" -ForegroundColor Cyan
    for ($i = 0; $i -lt $candidates.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $candidates[$i])
    }
    Write-Host ("  {0}) Enter a .duckdb path" -f ($candidates.Count + 1))
    Write-Host ("  {0}) Skip for now (set MARKET_DATA_DB later)" -f ($candidates.Count + 2))

    $selection = Read-Host "Select [$(if ($candidates.Count -gt 0) {1} else {$candidates.Count + 1})]"
    if (-not $selection) { $selection = if ($candidates.Count -gt 0) { "1" } else { "$($candidates.Count + 1)" } }
    $choice = 0
    if (-not [int]::TryParse($selection, [ref]$choice)) { return $null }
    if ($choice -ge 1 -and $choice -le $candidates.Count) { return $candidates[$choice - 1] }
    if ($choice -eq $candidates.Count + 1) {
        $entered = Read-Host "Full path to the Market Data Hub .duckdb file"
        if ($entered) { return [IO.Path]::GetFullPath($entered) }
        return $null
    }
    return $null
}

$ResolvedDb = Select-MarketDataDb $DbPath $SiblingHub
if ($ResolvedDb) {
    [Environment]::SetEnvironmentVariable("MARKET_DATA_DB", $ResolvedDb, "User")
    $env:MARKET_DATA_DB = $ResolvedDb
    Write-Host ""
    Write-Host "MARKET_DATA_DB set (persisted for your user account):" -ForegroundColor Green
    Write-Host "  $ResolvedDb"
    if (-not (Test-Path $ResolvedDb -PathType Leaf)) {
        Write-Host "  (file does not exist yet - market-data-hub will create it on first run)" -ForegroundColor DarkGray
    }
} else {
    Write-Host ""
    Write-Host "No database configured. Set it later with:" -ForegroundColor Yellow
    Write-Host '  [Environment]::SetEnvironmentVariable("MARKET_DATA_DB", "<path-to.duckdb>", "User")'
}

# ---------------------------------------------------------------------------
# 6. Verify + summary
# ---------------------------------------------------------------------------
Invoke-Step "Verifying imports" {
    & $Python -c "import lazyportfolio; import market_data_hub; print('LazyPortfolio environment OK')"
}

if ($HaveDev -and -not $SkipTests) {
    Invoke-Step "Running the test suite (this includes SLSQP solves and can take ~20 min)" {
        & $Python -m pytest -q
    }
}

Write-Host ""
Write-Host "==> Setup complete." -ForegroundColor Green
Write-Host "Run Tree Studio with:"
Write-Host "  $Python project\tree_studio.py 8766"
if ($HaveNode) {
    Write-Host "Run Tree Studio's JS contract tests with:"
    Write-Host "  npm test"
}
