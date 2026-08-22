$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonExecutable = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
} else {
    "python"
}

Push-Location $projectRoot
try {
    & $pythonExecutable -m unittest discover -s tests -v
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
