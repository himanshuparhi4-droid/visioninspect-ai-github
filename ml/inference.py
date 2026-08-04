from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ml.baseline_detector import (
    anomaly_mask,
    anomaly_score,
    embedding_anomaly_evidence,
    heatmap_overlay,
    load_reference_profile,
    normalized_anomaly_map,
    preprocess_gray,
)
from ml.padim_detector import PadimInferenceError, predict_with_anomaly_model


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceConfig:
    category: str
    anomaly_model_kind: str
    use_padim_inference: bool
    padim_inference_accelerator: str
    model_checkpoint_path: Path
    classifier_model_path: Path
    model_metadata_path: Path
    baseline_profile_path: Path
    baseline_threshold: float
    baseline_residual_threshold: float
    padim_score_threshold: float
    review_severity_threshold: float
    fail_severity_threshold: float
    critical_zones: tuple[str, ...] = ()
    cnn_classifier_model_path: Path | None = None
    openvino_path: Path | None = None
    openvino_calibrator_path: Path | None = None
    compact_classifier_path: Path | None = None
    use_openvino_inference: bool = False
    openvino_inference_device: str = "CPU"


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
def load_normal_profile(path_value: str) -> dict:
    try:
        return load_reference_profile(path_value)
    except FileNotFoundError as exc:
        raise InferenceError(str(exc)) from exc


def detection_confidence_from_score(score: float, threshold: float) -> float:
    """Estimate detector certainty from distance to its calibrated threshold."""
    margin = abs(score - threshold) / max(threshold, 1)
    return round(float(min(0.98, max(0.55, 0.55 + margin))), 4)


def has_localized_baseline_defect(score: float, baseline_threshold: float, binary_mask: np.ndarray) -> bool:
    """Require both an elevated score and a spatially meaningful residual."""
    return bool(score > baseline_threshold and np.any(binary_mask))


def compute_defect_geometry(
    binary_mask: np.ndarray,
    predicted_defect_type: str,
    critical_zones: tuple[str, ...] = (),
) -> dict:
    if predicted_defect_type == "good" or not np.any(binary_mask):
        return {
            "area_ratio": 0.0,
            "defect_center_x_ratio": None,
            "defect_center_y_ratio": None,
            "is_critical_location": False,
            "detected_regions": [],
            "critical_zones": list(critical_zones),
        }

    mask = np.asarray(binary_mask, dtype=bool)
    height, width = mask.shape[:2]
    ys, xs = np.where(mask)
    center_x_ratio = float(np.mean(xs) / max(width - 1, 1))
    center_y_ratio = float(np.mean(ys) / max(binary_mask.shape[0] - 1, 1))
    edge_width = max(1, int(round(min(height, width) * 0.15)))
    center_y_slice = slice(int(height * 0.25), max(int(height * 0.75), 1))
    center_x_slice = slice(int(width * 0.25), max(int(width * 0.75), 1))
    region_masks = {
        "center": np.zeros_like(mask),
        "edge": np.zeros_like(mask),
        "top": np.zeros_like(mask),
        "bottom": np.zeros_like(mask),
        "left": np.zeros_like(mask),
        "right": np.zeros_like(mask),
    }
    region_masks["center"][center_y_slice, center_x_slice] = True
    region_masks["edge"][:edge_width, :] = True
    region_masks["edge"][-edge_width:, :] = True
    region_masks["edge"][:, :edge_width] = True
    region_masks["edge"][:, -edge_width:] = True
    region_masks["top"][: height // 2, :] = True
    region_masks["bottom"][height // 2 :, :] = True
    region_masks["left"][:, : width // 2] = True
    region_masks["right"][:, width // 2 :] = True

    detected_regions = [
        name
        for name, region_mask in region_masks.items()
        if np.logical_and(mask, region_mask).sum() / max(mask.sum(), 1) >= 0.10
    ]
    normalized_critical_zones = {
        str(zone).strip().lower().replace("-", "_").replace(" ", "_") for zone in critical_zones
    }
    is_critical = bool(
        normalized_critical_zones.intersection(detected_regions)
        or normalized_critical_zones.intersection({"all", "whole_product", "entire_product"})
    )
    return {
        "area_ratio": float(np.mean(mask)),
        "defect_center_x_ratio": center_x_ratio,
        "defect_center_y_ratio": center_y_ratio,
        "is_critical_location": is_critical,
        "detected_regions": detected_regions,
        "critical_zones": sorted(normalized_critical_zones),
    }


def baseline_anomaly_prediction(image_path: Path, image_bgr: np.ndarray, config: InferenceConfig) -> dict:
    if config.baseline_profile_path.exists():
        profile = load_normal_profile(str(config.baseline_profile_path))
        diff_map = normalized_anomaly_map(image_bgr, profile)
        residual_score = round(float(anomaly_score(diff_map, mask=profile["foreground_mask"])), 4)
        if "embedding_bank" in profile:
            embedding_score, global_features = embedding_anomaly_evidence(image_path, profile["embedding_bank"])
            score = round(embedding_score, 6)
            detector_kind = "resnet18_normal_memory"
        else:
            score = residual_score
            global_features = None
            detector_kind = "opencv_normal_profile"
    else:
        raise InferenceError(f"Portable normal profile not found: {config.baseline_profile_path}")

    binary_mask = anomaly_mask(diff_map, config.baseline_residual_threshold)
    is_defective = has_localized_baseline_defect(score, config.baseline_threshold, binary_mask)
    detection_confidence = detection_confidence_from_score(score, config.baseline_threshold)

    return {
        "engine": "baseline",
        "detector_kind": detector_kind,
        "anomaly_score": score,
        "residual_score": residual_score,
        "decision_threshold": config.baseline_threshold,
        "is_defective": is_defective,
        "detection_confidence": detection_confidence,
        "anomaly_map": diff_map,
        "pred_mask": binary_mask,
        "global_features": global_features,
        "fallback_used": False,
        "fallback_reason": None,
    }


def live_anomaly_prediction(image_path: Path, image_bgr: np.ndarray, config: InferenceConfig) -> dict:
    if config.use_padim_inference or config.use_openvino_inference:
        try:
            return predict_with_anomaly_model(
                image_path,
                config.model_checkpoint_path,
                model_kind=config.anomaly_model_kind,
                score_threshold=config.padim_score_threshold,
                accelerator=config.padim_inference_accelerator,
                openvino_path=config.openvino_path if config.use_openvino_inference else None,
                openvino_calibrator_path=config.openvino_calibrator_path
                if config.use_openvino_inference
                else None,
                openvino_device=config.openvino_inference_device,
            )
        except PadimInferenceError as exc:
            fallback = baseline_anomaly_prediction(image_path, image_bgr, config)
            fallback["fallback_used"] = True
            fallback["fallback_reason"] = str(exc)
            return fallback

    return baseline_anomaly_prediction(image_path, image_bgr, config)


def classify_prediction(
    image_path: Path,
    score: float,
    detection_confidence: float,
    is_defective: bool,
    binary_mask: np.ndarray,
    config: InferenceConfig,
    global_features: np.ndarray | None = None,
) -> dict:
    if not is_defective:
        return {
            "defect_type": "good",
            "confidence": detection_confidence,
            "detection_confidence": detection_confidence,
            "classification_confidence": None,
            "classification_error": None,
            "class_probabilities": {"good": detection_confidence},
        }

    if not config.classifier_model_path.exists():
        return {
            "defect_type": "unknown_defect",
            "confidence": detection_confidence,
            "detection_confidence": detection_confidence,
            "classification_confidence": None,
            "classification_error": f"Classifier artifact not found: {config.classifier_model_path}",
            "class_probabilities": {"unknown_defect": detection_confidence},
        }

    try:
        from ml.defect_classifier import classify_defect_type

        classification = classify_defect_type(
            image_path,
            config.classifier_model_path,
            defect_mask=binary_mask,
            cnn_classifier_path=config.cnn_classifier_model_path,
            compact_classifier_path=config.compact_classifier_path,
            global_features=global_features,
        )
    except Exception as exc:
        return {
            "defect_type": "unknown_defect",
            "confidence": detection_confidence,
            "detection_confidence": detection_confidence,
            "classification_confidence": None,
            "classification_error": str(exc),
            "class_probabilities": {"unknown_defect": detection_confidence},
        }

    if classification["defect_type"] == "good":
        # The anomaly detector found a defect but the type classifier selected
        # good. Keep the anomaly evidence, but do not present the classifier's
        # high "good" probability as confidence in an unknown defect type.
        return {
            "defect_type": "unknown_defect",
            "confidence": detection_confidence,
            "detection_confidence": detection_confidence,
            "classification_confidence": None,
            "classification_error": "Defect detector and subtype classifier produced conflicting decisions",
            "class_probabilities": {
                "unknown_defect": detection_confidence,
                "good": round(1 - detection_confidence, 4),
            },
        }
    classification["detection_confidence"] = detection_confidence
    classification["classification_confidence"] = float(classification["confidence"])
    classification["classification_error"] = None
    return classification


def apply_quality_decision(severity: dict, config: InferenceConfig, *, is_defective: bool) -> dict:
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
    if is_defective:
        return {
            **severity,
            "pass_fail": "Review",
            "recommended_action": "An anomaly was detected; manual quality review is required before release",
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
    detection_confidence: float,
    classification_confidence: float | None,
    classification_error: str | None,
    anomaly_score_value: float,
    decision_threshold: float,
    geometry: dict,
    severity: dict,
    anomaly_map_value: np.ndarray,
    engine: str,
    fallback_used: bool,
    fallback_reason: str | None,
    decision_basis: str = "score_threshold",
    calibrated_defect_probability: float | None = None,
    calibration_threshold: float | None = None,
) -> dict:
    heatmap_intensity = float(np.percentile(anomaly_map_value, 95)) if anomaly_map_value.size else 0.0
    area_percent = float(geometry["area_ratio"] * 100)
    notes: list[str] = []
    calibrated_decision = decision_basis == "spatial_calibrator" and calibrated_defect_probability is not None
    if prediction == "Good":
        if calibrated_decision:
            notes.append(
                "Spatial anomaly calibration classified the image as normal "
                f"({calibrated_defect_probability * 100:.1f}% defect probability)."
            )
        elif anomaly_score_value <= decision_threshold:
            notes.append("Anomaly score stayed below the active decision threshold.")
        else:
            notes.append(
                "Anomaly score was borderline, but no localized defect region remained after residual-noise filtering."
            )
    else:
        if calibrated_decision:
            notes.append(
                "Spatial anomaly calibration classified the image as defective "
                f"({calibrated_defect_probability * 100:.1f}% defect probability)."
            )
        else:
            notes.append("Anomaly score exceeded the active decision threshold.")
        notes.append(f"Defect area covers approximately {area_percent:.2f}% of the inspected image.")
        if geometry.get("is_critical_location"):
            regions = ", ".join(geometry.get("detected_regions", [])) or "configured"
            notes.append(f"Detected anomaly overlaps a configured critical zone ({regions}).")
        if classification_confidence is None:
            notes.append("Defect subtype could not be identified reliably; manual classification is required.")
        else:
            notes.append(f"Classifier selected '{defect_type}' with {confidence * 100:.1f}% confidence.")
    if classification_error:
        notes.append(f"Classifier fallback reason: {classification_error}")
    if fallback_used:
        notes.append(f"Fallback inference was used: {fallback_reason}")

    return {
        "engine": engine,
        "active_inference_engine": engine,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "decision_threshold": round(float(decision_threshold), 4),
        "decision_basis": decision_basis,
        "calibrated_defect_probability": (
            round(float(calibrated_defect_probability), 4)
            if calibrated_defect_probability is not None
            else None
        ),
        "calibration_threshold": (
            round(float(calibration_threshold), 4) if calibration_threshold is not None else None
        ),
        "anomaly_score": round(float(anomaly_score_value), 4),
        "detection_confidence": round(float(detection_confidence), 4),
        "classification_confidence": (
            round(float(classification_confidence), 4) if classification_confidence is not None else None
        ),
        "heatmap_intensity_p95": round(float(heatmap_intensity), 4),
        "defect_area_percent": round(area_percent, 2),
        "critical_location": bool(geometry.get("is_critical_location")),
        "critical_zones": geometry.get("critical_zones", []),
        "detected_regions": geometry.get("detected_regions", []),
        "defect_center_x_ratio": geometry.get("defect_center_x_ratio"),
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

    classification = classify_prediction(
        image_path,
        score,
        detection_confidence,
        is_defective,
        binary_mask,
        config,
        global_features=anomaly.get("global_features"),
    )
    confidence = float(classification["confidence"])
    defect_type = classification["defect_type"]
    prediction = "Defective" if is_defective else "Good"
    geometry = compute_defect_geometry(binary_mask, defect_type, config.critical_zones)
    severity = calculate_severity_from_prediction(
        defect_type=defect_type,
        confidence=detection_confidence,
        area_ratio=geometry["area_ratio"],
        is_critical_location=geometry["is_critical_location"],
        defect_center_y_ratio=geometry["defect_center_y_ratio"],
    )
    severity = apply_quality_decision(severity, config, is_defective=is_defective)
    metadata = load_model_metadata(str(config.model_metadata_path))
    model_name = str(metadata.get("model_name", "model"))
    model_version = str(metadata.get("model_version", "local"))
    detector_kind = str(anomaly.get("detector_kind", anomaly["engine"]))
    model_used = (
        f"{model_name}:{model_version} ({anomaly['engine']})"
        if anomaly["engine"] != "baseline"
        else f"portable-baseline:{config.category} ({detector_kind})"
    )
    explainability = build_explainability(
        prediction=prediction,
        defect_type=defect_type,
        confidence=confidence,
        detection_confidence=detection_confidence,
        classification_confidence=classification.get("classification_confidence"),
        classification_error=classification.get("classification_error"),
        anomaly_score_value=score,
        decision_threshold=float(anomaly.get("decision_threshold", 0.5)),
        geometry=geometry,
        severity=severity,
        anomaly_map_value=diff_map,
        engine=anomaly["engine"],
        fallback_used=bool(anomaly.get("fallback_used", False)),
        fallback_reason=anomaly.get("fallback_reason"),
        decision_basis=str(anomaly.get("decision_basis", "score_threshold")),
        calibrated_defect_probability=anomaly.get("calibrated_defect_probability"),
        calibration_threshold=anomaly.get("calibration_threshold"),
    )

    processed = np.clip(preprocess_gray(image_bgr), 0, 255).astype(np.uint8)
    heatmap = heatmap_overlay(image_bgr, diff_map, binary_mask=binary_mask)

    return {
        "model_category": config.category,
        "prediction": prediction,
        "defect_type": defect_type,
        "confidence": confidence,
        "detection_confidence": detection_confidence,
        "classification_confidence": classification.get("classification_confidence"),
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
        "model_version": model_used,
        "model_used": model_used,
        "active_inference_engine": anomaly["engine"],
        "fallback_used": bool(anomaly.get("fallback_used", False)),
        "fallback_reason": anomaly.get("fallback_reason"),
        "processed_image": processed,
        "heatmap_image": heatmap,
        "anomaly_map": diff_map,
        "pred_mask": binary_mask,
    }
