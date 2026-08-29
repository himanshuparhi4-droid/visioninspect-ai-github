# VisionInspect AI

VisionInspect AI is a full-stack manufacturing quality inspection platform. It accepts product images, detects visual anomalies, classifies defect types, produces heatmaps, calculates severity, records pass/review/fail decisions, and supports manual review, rework, reports, analytics, users, and audit history.

The repository is designed to run locally after cloning. The website does not require the full MVTec AD dataset, MongoDB Atlas, Cloudinary, EC2, or S3.

## Main Features

- Login, registration, admin approval, roles, and JWT authentication.
- Manual image inspection and batches of up to 20 images.
- Camera simulation using bundled bottle sample frames.
- Fifteen MVTec product categories available for manual upload.
- Category-specific OpenVINO anomaly detection for all 15 supported products.
- OpenCV heatmap localization and category-specific defect subtype classification.
- Lazy model loading with a bounded cache for low-memory cloud instances.
- Optional full PaDiM/PatchCore checkpoints for local retraining and accelerator-backed experiments.
- Weighted severity scoring and pass/review/fail decisions.
- Inspection history, reviewer actions, rework tickets, and audit logs.
- PDF reports and production-quality analytics.
- Local file storage by default, with optional Cloudinary support.
- Local MongoDB by default, with optional MongoDB Atlas support.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 16, React 19, CSS, Lucide icons |
| Backend | FastAPI, Pydantic, Beanie, PyMongo Async |
| Database | MongoDB |
| Image processing | OpenCV, NumPy |
| Classification | PyTorch, Torchvision ResNet18, scikit-learn |
| Anomaly detection | OpenVINO Runtime, Anomalib PaDiM/PatchCore training pipeline |
| Reports | ReportLab |
| Quality | pytest, Ruff, Black, Prettier, Playwright |

## Runtime Modes

| Mode | Included after clone | Purpose |
| --- | --- | --- |
| OpenVINO anomaly runtime | Yes | Active, calibrated Good/Defective detection for all 15 categories |
| Defect subtype runtime | Yes | Category-specific ONNX CNN, compact forest, or scikit-learn classifier |
| Portable fallback | Yes | Normal-memory detector and OpenCV localization if an OpenVINO artifact cannot load |
| PaDiM/PatchCore checkpoints | Optional | Retraining, benchmarking, and accelerator-backed local experiments |

OpenVINO CPU inference is the default website runtime. Models are loaded only for the selected category and retained in a bounded cache. Every response identifies the active engine, model version, and any fallback reason. The portable path is a real fail-safe runtime, not a mock result.

## Included Model Assets

The repository keeps runtime assets small:

- OpenVINO model exports and spatial calibrators for all 15 categories.
- Category normal-image profiles, model metadata, and cross-validated subtype classifiers.
- SHA-256 integrity information exposed by the model registry for every portable artifact.
- Shared ONNX feature-extractor weights used by portable fallback and selected subtype models.
- Twelve bundled bottle camera-simulation frames.

The following large or licensed assets are intentionally excluded:

- Full MVTec AD dataset.
- Training checkpoints.
- Training outputs and temporary reports.

## Project Structure

```text
visioninspect-ai/
|-- backend/                 FastAPI API, database models, routes, and services
|-- frontend/                Next.js application
|-- ml/                      Shared inference, preprocessing, classification, and severity logic
|-- models/
|   |-- categories/          Uniform per-category classifiers, calibrators, profiles, metadata, and optional checkpoints
|   |-- exported/            Deployable OpenVINO models for all 15 categories
|   |-- shared/              Shared feature-extraction runtime assets
|   |-- defect_classifier.pkl
|   `-- model_metadata.json
|-- notebooks/               AI/ML learning and experiment notebooks
|-- scripts/                 Training, calibration, seeding, and model utilities
|-- tests/                   Backend, ML, camera, and portability tests
|-- .env.example             Safe local configuration template
|-- pyproject.toml           Python test and formatting configuration
`-- README.md                Complete project guide
```

## Local Setup

Use Python 3.13.5, Node.js 20+, and a local MongoDB service. The repository pins this Python version for local and Render deployments.

Create a root `.env`:

```powershell
Copy-Item .env.example .env
```

Install the backend runtime:

```powershell
python -m pip install -r backend/requirements.txt
```

Install the frontend:

```powershell
cd frontend
npm ci
cd ..
```

Start MongoDB, then start the backend:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

No `frontend/.env.local` is required for the default localhost ports. To use a different API URL, create it with:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Open:

- Frontend: `http://localhost:3000`
- API health: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`

Default local administrator:

```text
Email: admin@visioninspect.ai
Password: Admin@12345
```

## Local Configuration

`.env.example` is safe to commit. `.env` is private and ignored by Git.

Important settings:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=visioninspect_ai

USE_PADIM_INFERENCE=false
USE_OPENVINO_INFERENCE=true
OPENVINO_INFERENCE_DEVICE=CPU

BOOTSTRAP_ADMIN_ENABLED=true
BOOTSTRAP_ADMIN_EMAIL=admin@visioninspect.ai
BOOTSTRAP_ADMIN_PASSWORD=Admin@12345
```

Cloudinary is optional. Empty Cloudinary values make the backend serve images, heatmaps, and reports from local storage.
Set `USE_CLOUDINARY_STORAGE=true` only when online media storage is intentionally required.

The bootstrap account is created only in development. Production mode never creates the documented local administrator automatically.

## Inspection Workflow

```text
Login
  -> Select product category and image
  -> Validate image and metadata
  -> Run calibrated category anomaly detector
  -> Generate anomaly map and heatmap
  -> Classify defect type
  -> Calculate severity
  -> Produce pass/review/fail decision
  -> Save inspection in MongoDB
  -> Review, report, rework, and analytics workflows
```

The backend automatically fills missing product, batch, line, shift, operator, and source metadata.

## Supported Categories

```text
bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
pill, screw, tile, toothbrush, transistor, wood, zipper
```

Manual and batch upload support all 15 categories through the bundled OpenVINO artifacts. Camera simulation shows only categories for which redistribution-safe sample frames are physically present; the repository currently includes bottle samples.

## Verified Model Capability

The table below is the release snapshot used by the website model-metrics page. Binary F1 measures Good/Defective detection. Subtype accuracy and subtype macro F1 measure only the harder defect-type classification task. A category is marked `Production` only when binary F1 is at least 90% and subtype macro F1 is at least 85%; otherwise subtype predictions are shown as AI suggestions and routed to manual review.

The subtype numbers below come from a full deployed Render API audit on 2026-08-29. The audit uploaded all 1,258 defective MVTec test images through `https://visioninspect-ai-backend.onrender.com/inspections/inspect`, so these are deployed-runtime results, not local-only notebook results.

| Category | Binary F1 | Deployed Subtype Accuracy | Deployed Subtype Macro F1 | Active Detector | Release Status |
| --- | ---: | ---: | ---: | --- | --- |
| Bottle | 95.08% | 90.48% | 90.37% | PaDiM OpenVINO | Production |
| Cable | 97.24% | 90.22% | 81.27% | PatchCore OpenVINO | Manual review |
| Capsule | 95.19% | 100.00% | 100.00% | PatchCore OpenVINO | Production |
| Carpet | 94.25% | 89.89% | 89.83% | PaDiM OpenVINO | Production |
| Grid | 93.81% | 100.00% | 100.00% | PatchCore OpenVINO | Production |
| Hazelnut | 97.14% | 64.29% | 59.20% | PatchCore OpenVINO | Manual review |
| Leather | 93.71% | 84.78% | 85.33% | PaDiM OpenVINO | Production |
| Metal nut | 92.74% | 76.34% | 76.08% | PaDiM OpenVINO | Manual review |
| Pill | 96.45% | 87.23% | 82.20% | PatchCore OpenVINO | Manual review |
| Screw | 95.32% | 100.00% | 100.00% | PatchCore OpenVINO | Production |
| Tile | 94.67% | 96.43% | 80.70% | PaDiM OpenVINO | Manual review |
| Toothbrush | 91.23% | 93.33% | 48.28% | PaDiM OpenVINO | Manual review |
| Transistor | 93.83% | 95.00% | 95.10% | PatchCore OpenVINO | Production |
| Wood | 94.02% | 100.00% | 100.00% | PaDiM OpenVINO | Production |
| Zipper | 95.69% | 93.28% | 84.07% | PatchCore OpenVINO | Manual review |

These metrics use the evaluation protocol recorded in each category's `model_metadata.json`. Binary calibration uses nested stratified validation or a separate calibration/holdout split. Subtype metrics use the exact deployed backend path with the active anomaly detector and classifier artifacts. MVTec AD has few labelled examples for several subtypes, so the project does not claim that all subtype models are production-ready. The weakest deployed subtype categories are currently `hazelnut`, `metal_nut`, `pill`, `tile`, `toothbrush`, and `zipper`.

## Advanced Model Setup

The active category registry selects PaDiM for `bottle`, `carpet`, `leather`,
`metal_nut`, `tile`, `toothbrush`, and `wood`. PatchCore is selected for `cable`,
`capsule`, `grid`, `hazelnut`, `pill`, `screw`, `transistor`, and `zipper`.
Their deployable OpenVINO exports are included and run on CPU. Full training checkpoints remain optional for local or accelerator-backed retraining.

Current category evaluations report image-level binary F1 of at least 91.23%
and AUROC of at least 95.00% across all 15 categories. Deployed subtype
classification is strongest for `bottle`, `capsule`, `carpet`, `grid`, `screw`,
`transistor`, and `wood`, while categories below the subtype release target are
kept in manual-review mode.
Defect-subtype classification is a separate, harder task and varies by category
because several MVTec defect subtypes contain only a small number of labelled examples.

Advanced models are optional. For a category, place its checkpoint in the path specified by `models/category_model_registry.json`, install the development/ML dependencies, and enable:

```env
USE_PADIM_INFERENCE=true
PADIM_INFERENCE_ACCELERATOR=auto
```

OpenVINO is enabled by default and can be configured with:

```env
USE_OPENVINO_INFERENCE=true
OPENVINO_INFERENCE_DEVICE=CPU
```

Use `CPU` on Render. If OpenVINO is intentionally disabled or an artifact fails validation, the response identifies the portable fallback and its reason.

The root development requirements include Anomalib and notebook tooling:

```powershell
python -m pip install -r requirements-dev.txt
```

## Tests And Quality Checks

Run Python tests:

```powershell
python -m pytest tests -q
```

Run Python linting:

```powershell
python -m ruff check backend ml scripts tests
```

Run the frontend build and formatting check:

```powershell
cd frontend
npm run lint
npm run format:check
npm run build
```

Optional browser smoke tests:

```powershell
cd frontend
npx playwright test
```

## Data And Storage

- MongoDB stores users, inspections, reports, rework tickets, model records, production metadata, and audit logs.
- Inspection and batch records do not expire automatically.
- Local uploads are retained until intentionally removed.
- Cloudinary can replace local media storage by setting its three credentials.

## Security Notes

- Never commit `.env`, MongoDB credentials, Cloudinary secrets, or production JWT secrets.
- Change the bootstrap password before sharing a persistent environment.
- Set `ENVIRONMENT=production` and a strong `SECRET_KEY` outside local development.
- Public registrations remain pending until an administrator approves them.

## Current Scope

- Camera acquisition is a simulation, not a direct industrial camera/PLC connection.
- MES, ERP, and PLC integrations are API-ready future integrations.
- The portable runtime is calibrated per category, while full PaDiM/PatchCore checkpoints remain available for higher-capacity local serving and future factory-specific tuning.
- Model quality varies by category and should be recalibrated on the target factory's own images before production use.
- The MVTec dataset is not redistributed by this repository.
