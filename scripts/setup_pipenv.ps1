param(
    [switch]$InstallPipenv = $false
)

$ErrorActionPreference = "Stop"

Write-Host "[pipenv] Project root:" (Get-Location).Path

if ($InstallPipenv) {
    Write-Host "[pipenv] Ensuring pipenv is installed..."
    python -m pip install --upgrade pip
    python -m pip install pipenv
}

if (-not (Get-Command pipenv -ErrorAction SilentlyContinue)) {
    throw "pipenv is not installed. Re-run with -InstallPipenv or install it manually."
}

Write-Host "[pipenv] Installing dependencies from Pipfile..."
pipenv install --dev

Write-Host "[pipenv] Environment ready. Activate with:`n`n    pipenv shell`n"

Write-Host "[pipenv] To run the pipeline inside the environment:`n"`
    pipenv run python main.py --data-path ..\Data\creditcard\creditcard.csv`n"

