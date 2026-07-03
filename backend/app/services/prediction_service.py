from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.config import settings
from app.services.cloudinary_service import upload_image_or_local_url
from app.services.padim_service import PadimInferenceError, predict_with_padim

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class PredictionError(RuntimeError):
    pass


def resolve_backend_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (BACKEND_DIR / path).resolve()


def uploads_path(*parts: str) -> Path:
    path = resolve_backend_path(settings.upload_dir).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def load_model_metadata() -> dict:
    path = resolve_backend_path(settings.model_metadata_path)
    if not path.exists():
        return {
            "model_name": "classifier",
            "model_version": "local",
            "metrics": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_reference_image() -> np.ndarray:
    path = resolve_backend_path(settings.baseline_reference_path)
    reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise PredictionError(f"Baseline reference image not found: {path}")
    return reference.astype(np.float32)


@lru_cache(maxsize=1)
def load_classifier_runtime() -> tuple[dict, object, object, object]:
    from ml.classifier import build_resnet18_feature_extractor, load_classifier_bundle

    path = resolve_backend_path(settings.classifier_model_path)
    if not path.exists():
        raise PredictionError(f"Classifier artifact not found: {path}")

    bundle = load_classifier_bundle(path)
    feature_extractor, preprocess, device = build_resnet18_feature_extractor()
    return bundle, feature_extractor, preprocess, device


def classify_defect_type(image_path: str | Path) -> dict:
    from ml.classifier import extract_features

    bundle, feature_extractor, preprocess, device = load_classifier_runtime()
    features = extract_features(
        [image_path],
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    classifier = bundle["classifier"]
    label = str(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    class_probabilities = {
        str(class_name): round(float(probability), 4)
        for class_name, probability in zip(classifier.classes_, probabilities)
    }

    return {
        "defect_type": label,
        "confidence": round(float(np.max(probabilities)), 4),
        "class_probabilities": class_probabilities,
    }


def baseline_fallback_classification(score: float) -> dict:
    from app.services.model_settings_service import load_runtime_settings

    threshold = load_runtime_settings().baseline_threshold
    is_defective = score > threshold
    margin = abs(score - threshold) / max(threshold, 1)
    confidence = round(float(min(0.98, max(0.55, 0.55 + margin))), 4)
    defect_type = "contamination" if is_defective else "good"
    return {
        "defect_type": defect_type,
        "confidence": confidence,
        "class_probabilities": {
            "good": round(1 - confidence, 4) if is_defective else confidence,
            "contamination": confidence if is_defective else round(1 - confidence, 4),
        },
    }


def compute_defect_geometry(binary_mask: np.ndarray, predicted_defect_type: str) -> dict:
    if predicted_defect_type == "good" or not np.any(binary_mask):
        return {
            "area_ratio": 0.0,
            "defect_center_y_ratio": None,
            "is_critical_location": False,
        }

    ys, _ = np.where(binary_mask)
    center_y_ratio = float(np.mean(ys) / max(binary_mask.shape[0] - 1, 1))
    return {
        "area_ratio": float(np.mean(binary_mask)),
        "defect_center_y_ratio": center_y_ratio,
        "is_critical_location": 0.30 <= center_y_ratio <= 0.70,
    }


def save_visual_outputs(image_bgr: np.ndarray, diff_map: np.ndarray) -> dict:
    from ml.baseline_detector import heatmap_overlay, preprocess_gray

    output_stem = uuid4().hex

    processed_path = uploads_path("processed").joinpath(f"{output_stem}_processed.png")
    processed = preprocess_gray(image_bgr)
    cv2.imwrite(str(processed_path), np.clip(processed, 0, 255).astype(np.uint8))

    heatmap_path = uploads_path("heatmaps").joinpath(f"{output_stem}_heatmap.png")
    cv2.imwrite(str(heatmap_path), heatmap_overlay(image_bgr, diff_map))

    return {
        "processed_image_path": str(processed_path),
        "processed_image_url": upload_image_or_local_url(processed_path, "processed"),
        "heatmap_path": str(heatmap_path),
        "heatmap_url": upload_image_or_local_url(heatmap_path, "heatmaps"),
    }


def baseline_anomaly_prediction(image_bgr: np.ndarray) -> dict:
    from ml.baseline_detector import anomaly_map, anomaly_score
    from app.services.model_settings_service import load_runtime_settings

    reference = load_reference_image()
    threshold = load_runtime_settings().baseline_threshold
    diff_map = anomaly_map(image_bgr, reference)
    score = round(float(anomaly_score(diff_map)), 4)
    binary_mask = diff_map > threshold

    return {
        "engine": "baseline",
        "anomaly_score": score,
        "decision_threshold": threshold,
        "is_defective": score > threshold,
        "detection_confidence": baseline_fallback_classification(score)["confidence"],
        "anomaly_map": diff_map,
        "pred_mask": binary_mask,
    }


def build_explainability(
    *,
    prediction: str,
    defect_type: str,
    confidence: float,
    anomaly_score: float,
    decision_threshold: float,
    geometry: dict,
    severity: dict,
    anomaly_map: np.ndarray,
    engine: str,
) -> dict:
    heatmap_intensity = float(np.percentile(anomaly_map, 95)) if anomaly_map.size else 0.0
    area_percent = float(geometry["area_ratio"] * 100)
    notes: list[str] = []
    if prediction == "Good":
        notes.append("Anomaly score stayed below the active decision threshold.")
    else:
        notes.append("Anomaly score exceeded the active decision threshold.")
        notes.append(f"Defect area covers approximately {area_percent:.2f}% of the inspected image.")
        if geometry.get("is_critical_location"):
            notes.append("Detected anomaly is located in a critical center-region heuristic zone.")
        notes.append(f"Classifier selected '{defect_type}' with {confidence * 100:.1f}% confidence.")

    return {
        "engine": engine,
        "decision_threshold": round(float(decision_threshold), 4),
        "anomaly_score": round(float(anomaly_score), 4),
        "heatmap_intensity_p95": round(float(heatmap_intensity), 4),
        "defect_area_percent": round(area_percent, 2),
        "critical_location": bool(geometry.get("is_critical_location")),
        "defect_center_y_ratio": geometry.get("defect_center_y_ratio"),
        "severity_basis": severity.get("components", {}),
        "notes": notes,
    }


def live_anomaly_prediction(image_path: Path, image_bgr: np.ndarray) -> dict:
    if settings.use_padim_inference:
        checkpoint_path = resolve_backend_path(settings.model_checkpoint_path)
        try:
            return predict_with_padim(image_path, checkpoint_path)
        except PadimInferenceError:
            pass

    return baseline_anomaly_prediction(image_bgr)


def inspect_image_file(image_path: str | Path) -> dict:
    from ml.severity import calculate_severity_from_prediction
    from app.services.model_settings_service import load_runtime_settings

    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise PredictionError(f"Could not read image: {image_path}")

    anomaly = live_anomaly_prediction(image_path, image_bgr)
    diff_map = anomaly["anomaly_map"]
    binary_mask = anomaly["pred_mask"]
    score = float(anomaly["anomaly_score"])
    is_defective = bool(anomaly["is_defective"])
    detection_confidence = float(anomaly["detection_confidence"])

    if is_defective:
        try:
            classification = classify_defect_type(image_path)
        except Exception:
            classification = baseline_fallback_classification(score)

        if classification["defect_type"] == "good":
            classification["defect_type"] = "unknown_defect"
        confidence = max(float(classification["confidence"]), detection_confidence)
    else:
        classification = {
            "defect_type": "good",
            "confidence": detection_confidence,
            "class_probabilities": {"good": detection_confidence},
        }
        confidence = detection_confidence

    defect_type = classification["defect_type"]
    prediction = "Defective" if is_defective else "Good"
    geometry = compute_defect_geometry(binary_mask, defect_type)
    severity = calculate_severity_from_prediction(
        defect_type=defect_type,
        confidence=confidence,
        area_ratio=geometry["area_ratio"],
        is_critical_location=geometry["is_critical_location"],
        defect_center_y_ratio=geometry["defect_center_y_ratio"],
    )
    runtime_settings = load_runtime_settings()
    if severity["severity_score"] >= runtime_settings.fail_severity_threshold:
        severity["pass_fail"] = "Fail"
        severity["recommended_action"] = "Reject product or send to rework based on QA policy"
    elif severity["severity_score"] >= runtime_settings.review_severity_threshold:
        severity["pass_fail"] = "Review"
        severity["recommended_action"] = "Manual quality review required before release"
    else:
        severity["pass_fail"] = "Pass"
        severity["recommended_action"] = "Product generally acceptable"
    outputs = save_visual_outputs(image_bgr, diff_map)
    metadata = load_model_metadata()
    explainability = build_explainability(
        prediction=prediction,
        defect_type=defect_type,
        confidence=confidence,
        anomaly_score=score,
        decision_threshold=float(anomaly.get("decision_threshold", 0.5)),
        geometry=geometry,
        severity=severity,
        anomaly_map=diff_map,
        engine=anomaly["engine"],
    )

    return {
        "prediction": prediction,
        "defect_type": defect_type,
        "confidence": confidence,
        "anomaly_score": score,
        "defect_area_ratio": round(float(geometry["area_ratio"]), 4),
        "class_probabilities": classification["class_probabilities"],
        "severity_score": severity["severity_score"],
        "severity_level": severity["severity_level"],
        "severity_components": severity["components"],
        "explainability": explainability,
        "pass_fail": severity["pass_fail"],
        "recommended_action": severity["recommended_action"],
        "model_version": f"{metadata.get('model_name', 'model')}:{metadata.get('model_version', 'local')} ({anomaly['engine']})",
        **outputs,
    }


__all__ = ["PredictionError", "inspect_image_file"]
