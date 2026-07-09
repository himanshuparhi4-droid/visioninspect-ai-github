$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendRoot = Join-Path $ProjectRoot "backend"
$CheckpointPath = Join-Path $ProjectRoot "models\checkpoints\padim_mvtec_bottle_v1.ckpt"

if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    throw "PaDiM checkpoint not found: $CheckpointPath"
}

$env:USE_PADIM_INFERENCE = "true"
$env:MODEL_CHECKPOINT_PATH = $CheckpointPath
$env:PADIM_INFERENCE_ACCELERATOR = "auto"
$env:BACKEND_URL = "http://127.0.0.1:8000"
$env:FRONTEND_URL = "http://127.0.0.1:3000"
$env:CORS_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"

Set-Location $BackendRoot
python -m uvicorn main:app --host 127.0.0.1 --port 8000
