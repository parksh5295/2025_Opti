param(
    [string]$EnvDir = ".venv",
    [switch]$Recreate = $false
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] Project root:" (Get-Location).Path
Write-Host "[setup] Target virtual environment directory:" $EnvDir

if ($Recreate -and (Test-Path $EnvDir)) {
    Write-Host "[setup] Removing existing environment..."
    Remove-Item -Recurse -Force $EnvDir
}

if (-not (Test-Path $EnvDir)) {
    Write-Host "[setup] Creating virtual environment..."
    python -m venv $EnvDir
} else {
    Write-Host "[setup] Reusing existing environment. Use -Recreate to rebuild."
}

$activateScript = Join-Path $EnvDir "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    throw "Activation script not found: $activateScript"
}

Write-Host "[setup] Activating environment..."
. $activateScript

Write-Host "[setup] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

$requirementsPath = Join-Path (Get-Location) "requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Host "[setup] Installing dependencies from requirements.txt..."
    python -m pip install -r $requirementsPath
} else {
    Write-Warning "requirements.txt not found; skipping dependency installation."
}

Write-Host "[setup] Environment ready. To activate later, run:`n`n    . $activateScript`n"

