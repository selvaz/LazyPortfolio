# LazyPortfolio rolling-vs-expanding backtest + Telegram report wrapper
# Requires environment variables:
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID
#
# Windows Task Scheduler (even LogonType=Interactive) does not reliably
# inherit User-level environment variables set after the interactive
# session started -- same pattern as market-data-hub's
# run_regime_daily_with_telegram.ps1 and LazyRay's
# run_dalio_v2_with_telegram.ps1, both already proven in production on
# this machine. Without this explicit reload, the scheduled backtest
# would still complete and save to run_history, but the automatic
# Telegram delivery could silently fail to find the bot credentials.

param(
    [string[]]$RunBacktestArgs = @()
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Root
$Python = 'C:\ProgramData\spyder-6\python.exe'

Set-Location $RepoRoot

function Import-PersistedEnvVar($Name) {
    if (Test-Path "Env:$Name") {
        return
    }
    $value = [Environment]::GetEnvironmentVariable($Name, "User")
    if (!$value) {
        $value = [Environment]::GetEnvironmentVariable($Name, "Machine")
    }
    if ($value) {
        Set-Item -Path "Env:$Name" -Value $value
        Write-Host "[$(Get-Date -Format s)] Loaded $Name from persisted environment."
    }
}

Import-PersistedEnvVar "TELEGRAM_BOT_TOKEN"
Import-PersistedEnvVar "TELEGRAM_CHAT_ID"

Write-Host "[$(Get-Date -Format s)] Starting rolling vs expanding backtest: $($RunBacktestArgs -join ' ')"
& $Python (Join-Path $Root 'rolling_vs_expanding_backtest.py') @RunBacktestArgs
$exitCode = $LASTEXITCODE
Write-Host "[$(Get-Date -Format s)] rolling_vs_expanding_backtest.py exit code: $exitCode"

exit $exitCode
