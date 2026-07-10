from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ml.baseline_detector import anomaly_map, anomaly_score, heatmap_overlay, preprocess_gray
from ml.padim_detector import PadimInferenceError, predict_with_padim


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceConfig:
    use_padim_inference: bool
    padim_inference_accelerator: str
    model_checkpoint_path: Path
    classifier_model_path: Path
    model_metadata_path: Path
    baseline_reference_path: Path
    baseline_threshold: float
    padim_score_threshold: float
    review_severity_threshold: float
    fail_severity_threshold: float


def default_model_metadata() -> dict:
    return {
        "model_name": "classifier",
        "model_version": "local",
        "metrics": {},
    }


@lru_cache(maxsize=4)
def load_model_metadata(path_value: str) -> dict:
    path = Path(path_value)
    if not path.exists():
        return default_model_metadata()
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_reference_image(path_value: str) -> np.ndarray:
    path = Path(path_value)
    reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise InferenceError(f"Baseline reference image not found: {path}")
    return reference.astype(np.float32)


def baseline_fallback_classification(score: float, baseline_threshold: float) -> dict:
    is_defective = score > baseline_threshold
    margin = abs(score - baseline_threshold) / max(baseline_threshold, 1)
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


def baseline_anomaly_prediction(image_bgr: np.ndarray, config: InferenceConfig) -> dict:
    reference = load_reference_image(str(config.baseline_reference_path))
    diff_map = anomaly_map(image_bgr, reference)
    score = round(float(anomaly_score(diff_map)), 4)
    binary_mask = diff_map > config.baseline_threshold
    classification = baseline_fallback_classification(score, config.baseline_threshold)

    return {
        "engine": "baseline",
        "anomaly_score": score,
        "decision_threshold": config.baseline_threshold,
        "is_defective": score > config.baseline_threshold,
        "detection_confidence": classification["confidence"],
        "anomaly_map": diff_map,
        "pred_mask": binary_mask,
        "fallback_used": False,
        "fallback_reason": None,
    }


def live_anomaly_prediction(image_path: Path, image_bgr: np.ndarray, config: InferenceConfig) -> dict:
    if config.use_padim_inference:
        try:
            return predict_with_padim(
                image_path,
                config.model_checkpoint_path,
                score_threshold=config.padim_score_threshold,
                accelerator=config.padim_inference_accelerator,
            )
        except PadimInferenceError as exc:
            fallback = baseline_anomaly_prediction(image_bgr, config)
            fallback["fallback_used"] = True
            fallback["fallback_reason"] = str(exc)
            return fallback

    return baseline_anomaly_prediction(image_bgr, config)


def classify_prediction(image_path: Path, score: float, detection_confidence: float, is_defective: bool, config: InferenceConfig) -> dict:
    if not is_defective:
        return {
            "defect_type": "good",
            "confidence": detection_confidence,
            "class_probabilities": {"good": detection_confidence},
        }

    try:
        from ml.defect_classifier import classify_defect_type

        classification = classify_defect_type(image_path, config.classifier_model_path)
    except Exception:
        classification = baseline_fallback_classification(score, config.baseline_threshold)

    if classification["defect_type"] == "good":
        classification["defect_type"] = "unknown_defect"
    classification["confidence"] = max(float(classification["confidence"]), detection_confidence)
    return classification


def apply_quality_decision(severity: dict, config: InferenceConfig) -> dict:
    score = severity["severity_score"]

    if score >= config.fail_severity_threshold:
        return {
            **severity,
            "pass_fail": "Fail",
            "recommended_action": "Reject product or send to rework based on QA policy",
        }
    if score >= config.review_severity_threshold:
        return {
            **severity,
            "pass_fail": "Review",
            "recommended_action": "Manual quality review required before release",
        }
    return {
        **severity,
        "pass_fail": "Pass",
        "recommended_action": "Product generally acceptable",
    }


def build_explainability(
    *,
    prediction: str,
    defect_type: str,
    confidence: float,
    anomaly_score_value: float,
    decision_threshold: float,
    geometry: dict,
    severity: dict,
    anomaly_map_value: np.ndarray,
    engine: str,
    fallback_used: bool,
    fallback_reason: str | None,
) -> dict:
    heatmap_intensity = float(np.percentile(anomaly_map_value, 95)) if anomaly_map_value.size else 0.0
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
    if fallback_used:
        notes.append(f"Fallback inference was used: {fallback_reason}")

    return {
        "engine": engine,
        "active_inference_engine": engine,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "decision_threshold": round(float(decision_threshold), 4),
        "anomaly_score": round(float(anomaly_score_value), 4),
        "heatmap_intensity_p95": round(float(heatmap_intensity), 4),
        "defect_area_percent": round(area_percent, 2),
        "critical_location": bool(geometry.get("is_critical_location")),
        "defect_center_y_ratio": geometry.get("defect_center_y_ratio"),
        "severity_basis": severity.get("components", {}),
        "notes": notes,
    }


def inspect_image(image_path: str | Path, config: InferenceConfig) -> dict:
    from ml.severity import calculate_severity_from_prediction

    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise InferenceError(f"Could not read image: {image_path}")

    anomaly = live_anomaly_prediction(image_path, image_bgr, config)
    diff_map = anomaly["anomaly_map"]
    binary_mask = anomaly["pred_mask"]
    score = float(anomaly["anomaly_score"])
    is_defective = bool(anomaly["is_defective"])
    detection_confidence = float(anomaly["detection_confidence"])

    classification = classify_prediction(image_path, score, detection_confidence, is_defective, config)
    confidence = float(classification["confidence"])
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
    severity = apply_quality_decision(severity, config)
    metadata = load_model_metadata(str(config.model_metadata_path))
    explainability = build_explainability(
        prediction=prediction,
        defect_type=defect_type,
        confidence=confidence,
        anomaly_score_value=score,
        decision_threshold=float(anomaly.get("decision_threshold", 0.5)),
        geometry=geometry,
        severity=severity,
        anomaly_map_value=diff_map,
        engine=anomaly["engine"],
        fallback_used=bool(anomaly.get("fallback_used", False)),
        fallback_reason=anomaly.get("fallback_reason"),
    )

    processed = np.clip(preprocess_gray(image_bgr), 0, 255).astype(np.uint8)
    heatmap = heatmap_overlay(image_bgr, diff_map)

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
        "decision": severity["pass_fail"],
        "recommended_action": severity["recommended_action"],
        "model_version": f"{metadata.get('model_name', 'model')}:{metadata.get('model_version', 'local')} ({anomaly['engine']})",
        "model_used": f"{metadata.get('model_name', 'model')}:{metadata.get('model_version', 'local')} ({anomaly['engine']})",
        "active_inference_engine": anomaly["engine"],
        "fallback_used": bool(anomaly.get("fallback_used", False)),
        "fallback_reason": anomaly.get("fallback_reason"),
        "processed_image": processed,
        "heatmap_image": heatmap,
        "anomaly_map": diff_map,
        "pred_mask": binary_mask,
    }
