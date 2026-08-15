# VisionInspect AI

VisionInspect AI is a full-stack manufacturing quality inspection platform. It accepts product images, detects visual anomalies, classifies defect types, produces heatmaps, calculates severity, records pass/review/fail decisions, and supports manual review, rework, reports, analytics, users, and audit history.

The repository is designed to run locally after cloning. The website does not require the full MVTec AD dataset, MongoDB Atlas, Cloudinary, EC2, or S3.

## Main Features

- Login, registration, admin approval, roles, and JWT authentication.
- Manual image inspection and batches of up to 20 images.
- Camera simulation using bundled bottle sample frames.
- Fifteen MVTec product categories available for manual upload.
- Portable ResNet18 normal-memory anomaly screening with OpenCV heatmaps.
- Category-specific ResNet18 and texture-based defect subtype classification.
- Optional PaDiM/PatchCore and OpenVINO runtime when advanced artifacts are installed.
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
| Advanced anomaly models | Anomalib PaDiM/PatchCore, optional OpenVINO |
| Reports | ReportLab |
| Quality | pytest, Ruff, Black, Prettier, Playwright |

## Runtime Modes

| Mode | Included after clone | Purpose |
| --- | --- | --- |
| Portable hybrid runtime | Yes | ResNet18 normal-memory detection and OpenCV heatmaps for all 15 categories |
| Defect classifier | Yes | Defect-only subtype prediction using ResNet18, gradient, and color features |
| PaDiM/PatchCore | Optional | Stronger learned anomaly detection when checkpoints are installed |
| OpenVINO | Optional | CPU-optimized serving when exported models are installed |

Portable mode is the default. It is intentional production-style fallback behavior, not a demo stub. Advanced inference is enabled only when explicitly configured and a valid model artifact exists.

## Included Model Assets

The repository keeps runtime assets small:

- Category normal-image profiles, ResNet18 memory banks, and model metadata.
- Cross-validated category defect-subtype classifier artifacts.
- SHA-256 integrity information exposed by the model registry for every portable artifact.
- ResNet18 feature-extractor weights.
- Twelve bundled bottle camera-simulation frames.

The following large or licensed assets are intentionally excluded:

- Full MVTec AD dataset.
- Training checkpoints.
- OpenVINO exports.
- Training outputs and temporary reports.

## Project Structure

```text
visioninspect-ai/
|-- backend/                 FastAPI API, database models, routes, and services
|-- frontend/                Next.js application
|-- ml/                      Shared inference, preprocessing, classification, and severity logic
|-- models/
|   |-- categories/          Compact runtime assets for 14 non-bottle categories
|   |-- inference/           Bottle profile and bundled ResNet18 weights
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
USE_OPENVINO_INFERENCE=false

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

Manual upload works from the bundled compact runtime assets. Camera simulation shows categories for which sample frames are physically available; the clone includes bottle samples.

## Advanced Model Setup

The active category registry selects PaDiM for `bottle`, `carpet`, `grid`,
`leather`, `metal_nut`, `tile`, `toothbrush`, and `wood`. PatchCore is selected
for `cable`, `capsule`, `hazelnut`, `pill`, `screw`, `transistor`, and `zipper`.
The portable CPU path uses the shared ONNX feature extractor and a calibrated
OpenVINO-compatible runtime for every category; full checkpoints remain optional
for local or accelerator-backed serving.

Current category evaluations exceed 90% image-level accuracy, F1, and AUROC for
Good/Defective detection in all 15 categories. Defect-subtype classification is
a separate, harder task and varies by category because several MVTec defect
subtypes contain only a small number of labeled examples.

Advanced models are optional. For a category, place its checkpoint in the path specified by `models/category_model_registry.json`, install the development/ML dependencies, and enable:

```env
USE_PADIM_INFERENCE=true
PADIM_INFERENCE_ACCELERATOR=auto
```

OpenVINO must be enabled separately:

```env
USE_OPENVINO_INFERENCE=true
OPENVINO_INFERENCE_DEVICE=CPU
```

Keeping OpenVINO disabled prevents unsupported GPU compilation from delaying or blocking an inspection.

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
