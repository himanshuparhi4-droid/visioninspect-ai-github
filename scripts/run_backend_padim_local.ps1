$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendRoot = Join-Path $ProjectRoot "backend"
$CheckpointPath = Join-Path $ProjectRoot "models\categories\bottle\padim_v1.ckpt"

if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    throw "PaDiM checkpoint not found: $CheckpointPath"
}

$env:USE_PADIM_INFERENCE = "true"
$env:PADIM_INFERENCE_ACCELERATOR = "auto"
$env:BACKEND_URL = "http://127.0.0.1:8000"
$env:FRONTEND_URL = "http://127.0.0.1:3000"
$env:CORS_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"

Set-Location $BackendRoot
python -m uvicorn main:app --host 127.0.0.1 --port 8000
