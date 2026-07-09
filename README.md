# VisionInspect AI

VisionInspect AI is a full-stack manufacturing defect detection and quality inspection platform. It lets a quality team upload or simulate product images, inspect them with computer vision, classify defects, calculate severity, generate reports, manage rework, and monitor production quality through dashboards.

## Core Features

- Authentication, registration, admin approval, and role-based access.
- Manual image inspection, batch inspection, and camera simulation.
- Image validation, preprocessing, anomaly detection, defect classification, and heatmap generation.
- Severity scoring using defect size, location, defect type, and model confidence.
- Pass, fail, review, approve, reject, and rework workflows.
- Rework tickets with priority, assignee, due date, aging, status, and resolution notes.
- PDF inspection reports with original image, heatmap, metadata, reviewer notes, and severity details.
- Analytics for inspection counts, defect trends, severity distribution, and line-wise quality.
- Model metrics page with classifier report, confusion matrix, threshold tuning, and calibration notes.
- MongoDB Atlas support for data storage and Cloudinary support for images, heatmaps, and reports.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | Next.js, React, CSS, Recharts, Lucide icons |
| Backend | FastAPI, Beanie, Motor, Pydantic |
| Database | MongoDB Atlas |
| Storage | Cloudinary |
| ML/CV | PyTorch, Anomalib PaDiM, OpenCV, scikit-learn, NumPy |
| Quality | pytest, Black, Ruff, Prettier |

## Project Structure

```text
visioninspect-ai/
|-- backend/        FastAPI app, routes, services, schemas, database models
|-- frontend/       Next.js dashboard and UI components
|-- ml/             image processing, classifier, severity, metrics, validation
|-- models/         model artifacts and metadata
|-- notebooks/      learning, dataset exploration, preprocessing, model experiments
|-- scripts/        seed and admin setup helpers
|-- tests/          backend, ML, and workflow tests
|-- deployment/     Render/Vercel deployment notes and config
|-- .env.example    environment variable template
|-- pyproject.toml  Python formatting and test config
`-- README.md
```

## Local Setup

Install backend and development dependencies:

```powershell
pip install -r backend/requirements.txt
pip install -r requirements-dev.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Create a root `.env` file. Use `.env.example` as the template:

```env
SECRET_KEY=replace-with-secure-secret
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

MONGODB_URI=your-mongodb-atlas-uri
MONGODB_DATABASE=visioninspect_ai

CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

USE_PADIM_INFERENCE=false
CLASSIFIER_MODEL_PATH=../models/defect_classifier.pkl
MODEL_METADATA_PATH=../models/model_metadata.json
BASELINE_REFERENCE_PATH=../models/inference/normal_reference.png
UPLOAD_DIR=app/uploads
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Start the backend:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open the app:

```text
http://127.0.0.1:3000
```

## Default Admin

```text
Email: admin@visioninspect.ai
Password: Admin@12345
```

Change this password before production use.

## Dataset

The project supports the MVTec AD bottle dataset locally:

```text
data/raw/mvtec_anomaly_detection/bottle/
```

The dataset is not committed to GitHub because it is large and has separate licensing terms. Camera simulation also includes bundled synthetic demo samples so the feature works without the full dataset.

## Local PaDiM Inference

The trained PaDiM checkpoint can be used locally without changing global system environment variables.

Run one image from the command line:

```powershell
python ml\predict.py --use-padim --image data\raw\mvtec_anomaly_detection\bottle\test\contamination\000.png
```

Start the backend with PaDiM enabled:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_backend_padim_local.ps1
```

Then check the backend health endpoint:

```text
http://127.0.0.1:8000/health
```

The response should show `padim_enabled: true` and `padim_checkpoint: true`.

## Model Deployment

The project currently supports:

- A lightweight live inference path using OpenCV baseline detection and the saved classifier artifact.
- PaDiM/Anomalib experiment support from the ML training workflow.
- Production PaDiM serving through a separate compute or model server when GPU-backed deployment is required.

For a dedicated PaDiM serving setup, host the checkpoint on the compute/model server and configure:

```env
USE_PADIM_INFERENCE=true
MODEL_CHECKPOINT_PATH=path-or-mounted-location-of-padim-checkpoint
```

Keep large checkpoints outside GitHub and manage them as deployment artifacts.

## Deployment

Recommended deployment split:

- Frontend: Vercel
- Backend: Render or a dedicated compute backend
- Database: MongoDB Atlas
- Storage: Cloudinary
- GPU/model serving: external compute server when PaDiM live inference is enabled

Deployment environment variables should match `.env.example`. On Vercel, set:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend-url
```

On the backend host, set MongoDB, Cloudinary, security, CORS, and model path variables.

## Quality Checks

Run backend and ML tests:

```powershell
pytest -q
```

Run frontend build:

```powershell
cd frontend
npm run build
```

Check frontend formatting:

```powershell
cd frontend
npm run format:check
```

Format frontend code:

```powershell
cd frontend
npm run format
```

Python formatting tools are configured in `pyproject.toml`:

```powershell
black backend ml tests scripts
ruff check backend ml tests scripts
```

## Screenshots

Add these screenshots after deployment:

| Page | Screenshot |
| --- | --- |
| Login and registration | `screenshots/login.png` |
| Dashboard analytics | `screenshots/dashboard.png` |
| Upload and inspection result | `screenshots/upload-result.png` |
| Heatmap and severity details | `screenshots/inspection-detail.png` |
| Rework queue and reports | `screenshots/rework-reports.png` |

## Known Limitations

- Industrial camera, PLC, MES, and ERP integrations are represented as simulation/API-ready workflows, not direct hardware integrations.
- Bundled camera samples are synthetic demo images; full dataset evaluation should use the local MVTec dataset.
- Large training datasets and heavyweight checkpoints are kept out of GitHub.
- Production PaDiM inference should be hosted on a dedicated compute/model server when GPU-backed serving is required.
