param(
    [int]$Port = 8766,
    [string]$Python = "py",
    [string]$DbPath,
    [string]$MarketDataHubPath,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $RepoRoot ".venv-tree-studio"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$SiblingHub = if ($MarketDataHubPath) {
    Resolve-Path $MarketDataHubPath -ErrorAction Stop
} else {
    Resolve-Path (Join-Path $RepoRoot "..\market-data-hub") -ErrorAction SilentlyContinue
}

function Select-MarketDataDb {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        return [IO.Path]::GetFullPath($RequestedPath)
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($env:MARKET_DATA_DB -and (Test-Path $env:MARKET_DATA_DB -PathType Leaf)) {
        [void]$candidates.Add([IO.Path]::GetFullPath($env:MARKET_DATA_DB))
    }
    if ($SiblingHub) {
        Get-ChildItem -LiteralPath $SiblingHub.Path -Filter "*.duckdb" -File -ErrorAction SilentlyContinue |
            ForEach-Object { [void]$candidates.Add($_.FullName) }
    }
    $candidates = @($candidates | Select-Object -Unique)

    Write-Host ""
    Write-Host "Market Data Hub database"
    for ($i = 0; $i -lt $candidates.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $candidates[$i])
    }
    Write-Host ("  {0}) Enter another .duckdb path" -f ($candidates.Count + 1))

    $defaultChoice = "1"
    $selection = Read-Host "Select database [$defaultChoice]"
    if (-not $selection) { $selection = $defaultChoice }
    $choice = 0
    if (-not [int]::TryParse($selection, [ref]$choice) -or $choice -lt 1 -or $choice -gt $candidates.Count + 1) {
        throw "Invalid database selection: $selection"
    }
    if ($choice -eq $candidates.Count + 1) {
        $entered = Read-Host "Full path to the Market Data Hub .duckdb file"
        if (-not $entered) { throw "A database path is required" }
        return [IO.Path]::GetFullPath($entered)
    }
    return $candidates[$choice - 1]
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Command
}

Invoke-Step "Creating virtual environment" {
    if (-not (Test-Path $VenvPython)) {
        if ($Python -eq "py") {
            Invoke-Native $Python @("-3.11", "-m", "venv", $VenvDir)
        } else {
            Invoke-Native $Python @("-m", "venv", $VenvDir)
        }
    } else {
        Write-Host "Reusing $VenvDir"
    }
}

Invoke-Step "Upgrading pip tooling" {
    Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
}

if ($SiblingHub) {
    Invoke-Step "Installing local market-data-hub checkout" {
        Invoke-Native $VenvPython @("-m", "pip", "install", "-e", $SiblingHub.Path)
    }
} else {
    Invoke-Step "Installing market-data-hub from GitHub" {
        Invoke-Native $VenvPython @("-m", "pip", "install", "market-data-hub @ git+https://github.com/selvaz/market-data-hub.git")
    }
}

Invoke-Step "Installing LazyPortfolio Tree Studio dependencies" {
    Invoke-Native $VenvPython @("-m", "pip", "install", "-e", "$RepoRoot")
}

Invoke-Step "Verifying imports" {
    Invoke-Native $VenvPython @("-c", "import lazyportfolio; import skfolio; import market_data_hub; print('LazyPortfolio Tree Studio environment OK')")
}

$DbPath = Select-MarketDataDb $DbPath
$env:MARKET_DATA_DB = $DbPath
Write-Host "Using MARKET_DATA_DB=$DbPath" -ForegroundColor DarkGray

if (-not (Test-Path $DbPath -PathType Leaf)) {
    if (-not $SiblingHub) {
        throw "Market Data Hub checkout not found. Pass -MarketDataHubPath or clone it next to LazyPortfolio."
    }
    $HubSetup = Join-Path $SiblingHub.Path "setup_first_run.ps1"
    if (-not (Test-Path $HubSetup -PathType Leaf)) {
        throw "Market Data Hub setup script not found: $HubSetup"
    }
    Invoke-Step "Running Market Data Hub first-run setup" {
        Invoke-Native "powershell.exe" @(
            "-ExecutionPolicy", "Bypass",
            "-File", $HubSetup,
            "-Python", $VenvPython,
            "-DbPath", $DbPath,
            "-SkipTests"
        )
    }
    if (-not (Test-Path $DbPath -PathType Leaf)) {
        throw "Market Data Hub setup completed without creating the selected database: $DbPath"
    }
} else {
    Write-Host "Reusing existing Market Data Hub database." -ForegroundColor Green
}

$Url = "http://127.0.0.1:$Port/"
Write-Host ""
Write-Host "LazyPortfolio Tree Studio will run at $Url" -ForegroundColor Green
Write-Host "Stop it with Ctrl+C in this terminal." -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Process $Url
}

Set-Location $RepoRoot
Invoke-Native $VenvPython @("project\tree_studio.py", "$Port")
