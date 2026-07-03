# VisionInspect AI

VisionInspect AI is a full-stack manufacturing defect detection and quality inspection system. It uses computer vision and AI to inspect product images, detect defects, classify defect type, calculate severity, generate reports, and show production analytics.

## Features

- Login, registration, user roles, and admin approval.
- Manual image upload for inspection.
- Camera simulation using MVTec bottle images.
- Batch image inspection.
- Defect detection with PaDiM support and OpenCV fallback.
- Defect classification: `good`, `broken_large`, `broken_small`, `contamination`.
- Severity score using defect size, location, defect type, and confidence.
- Pass, fail, review, approve, reject, and rework workflow.
- Rework ticket queue with assignment, priority, status, and resolution notes.
- Heatmap visualization.
- PDF inspection reports.
- Analytics dashboard.
- MongoDB Atlas database support.
- Cloudinary file storage support.
- Vercel frontend and Render backend deployment support.

## Tech Stack

- Frontend: Next.js, React, CSS
- Backend: FastAPI, Beanie, MongoDB
- Storage: Cloudinary
- Machine Learning: PyTorch, Anomalib PaDiM, OpenCV, scikit-learn, NumPy

## Main Structure

```text
backend/      FastAPI backend, routes, services, database models
frontend/     Next.js frontend dashboard
ml/           image processing, model logic, severity, validation
models/       saved lightweight model files and metadata
scripts/      setup and seed scripts
tests/        backend and ML tests
deployment/   deployment config
database/     MongoDB helper files
```

## Environment Setup

Create `.env` in the project root:

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
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Run Locally

Install backend dependencies:

```powershell
pip install -r backend/requirements.txt
pip install -r requirements-dev.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Start backend:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start frontend:

```powershell
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

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

The project uses the MVTec AD bottle dataset locally:

```text
data/raw/mvtec_anomaly_detection/bottle/
```

The dataset is not included in GitHub because it is large and has separate licensing terms.

## Tests

```powershell
pytest -q
```

Some dataset tests are skipped if the MVTec dataset is not present.

## Deployment

Recommended free deployment:

- Frontend: Vercel
- Backend: Render
- Database: MongoDB Atlas
- Storage: Cloudinary

For free deployment, keep:

```env
USE_PADIM_INFERENCE=false
```

The large PaDiM checkpoint is not included in GitHub.
