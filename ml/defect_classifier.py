from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
)


class DefectClassifierError(RuntimeError):
    pass


logger = logging.getLogger(__name__)
PORTABLE_CNN_CACHE_SIZE = max(1, min(int(os.getenv("CLASSIFIER_MODEL_CACHE_SIZE", "2")), 4))


class PortableCNNClassifier:
    """Thread-safe OpenCV DNN runtime for a fine-tuned CNN classifier."""

    def __init__(self, model_path: Path, metadata: dict):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.metadata = metadata
        self.lock = Lock()

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        with self.lock:
            self.net.setInput(tensor)
            return self.net.forward()


@lru_cache(maxsize=PORTABLE_CNN_CACHE_SIZE)
def load_portable_cnn_runtime(model_path_value: str, metadata_path_value: str) -> PortableCNNClassifier:
    model_path = Path(model_path_value)
    metadata_path = Path(metadata_path_value)
    if not model_path.exists() or not metadata_path.exists():
        raise DefectClassifierError(f"Portable CNN classifier artifacts are incomplete: {model_path.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("artifact_type") != "visioninspect_resnet18_finetuned_classifier_onnx":
        raise DefectClassifierError(f"Unsupported portable CNN classifier metadata: {metadata_path}")
    return PortableCNNClassifier(model_path, metadata)


def predict_portable_cnn_defect_type(
    image_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    *,
    defect_mask: np.ndarray | None = None,
) -> dict:
    from ml.object_preprocessing import prepare_classifier_view, read_bgr

    metadata_path = Path(metadata_path)
    model_path = Path(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    image_size = int(metadata["image_size"])
    crop_mode = str(metadata.get("preprocessing", {}).get("view", "defect"))
    image_bgr = read_bgr(image_path)
    if crop_mode == "full":
        from ml.object_preprocessing import enhance_contrast_bgr, resize_with_padding

        view = enhance_contrast_bgr(resize_with_padding(image_bgr, (image_size, image_size)))
    elif crop_mode == "object":
        view = prepare_classifier_view(image_bgr, None, image_size=image_size)
    elif crop_mode in {"defect", "object_crop_or_anomaly_mask_crop"}:
        view = prepare_classifier_view(image_bgr, defect_mask, image_size=image_size)
    elif crop_mode == "bbox":
        from ml.cnn_classifier import prepare_cnn_view
        view = prepare_cnn_view(image_bgr, defect_mask, image_size=image_size, crop_mode=crop_mode)
    else:
        raise DefectClassifierError(f"Unsupported portable CNN crop mode: {crop_mode}")

    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    tensor = np.transpose((rgb - mean) / std, (2, 0, 1))[None].astype(np.float32, copy=False)

    runtime = load_portable_cnn_runtime(str(model_path), str(metadata_path))
    logits = runtime.predict(tensor).reshape(-1)
    temperature = max(float(metadata.get("temperature", 1.0)), 0.05)
    logits = logits / temperature

    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    labels = [str(label) for label in metadata["labels"]]
    best_index = int(np.argmax(probabilities))
    return {
        "defect_type": labels[best_index],
        "confidence": round(float(probabilities[best_index]), 4),
        "class_probabilities": {
            label: round(float(probability), 4)
            for label, probability in zip(labels, probabilities, strict=True)
        },
        "classifier_engine": "fine_tuned_resnet18_onnx",
        "classifier_fallback_used": False,
        "classifier_fallback_reason": None,
        "confidence_calibrated": "temperature" in metadata,
    }


@lru_cache(maxsize=PORTABLE_CNN_CACHE_SIZE)
def load_classifier_runtime(path_value: str) -> dict:
    from ml.classifier import load_classifier_bundle

    path = Path(path_value)
    if not path.exists():
        raise DefectClassifierError(f"Classifier artifact not found: {path}")
    return load_classifier_bundle(path)


@lru_cache(maxsize=1)
def shared_feature_runtime():
    from ml.classifier import (
        DEFAULT_RESNET_ONNX_PATH,
        DEFAULT_RESNET_OPENVINO_PATH,
        build_opencv_resnet18_feature_extractor,
        build_openvino_resnet18_feature_extractor,
        build_resnet18_feature_extractor,
    )

    if DEFAULT_RESNET_OPENVINO_PATH.exists() and DEFAULT_RESNET_OPENVINO_PATH.with_suffix(".bin").exists():
        return build_openvino_resnet18_feature_extractor()
    if DEFAULT_RESNET_ONNX_PATH.exists():
        return build_opencv_resnet18_feature_extractor()
    return build_resnet18_feature_extractor()


def warm_shared_feature_runtime() -> None:
    """Load and execute the shared feature model before the first request."""
    from PIL import Image

    from ml.classifier import extract_pil_features

    feature_extractor, preprocess, device = shared_feature_runtime()
    extract_pil_features(
        [Image.new("RGB", (224, 224))],
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )


def release_classifier_runtimes() -> None:
    """Drop cached subtype models after constrained-host inference."""
    load_portable_cnn_runtime.cache_clear()
    load_classifier_runtime.cache_clear()
    shared_feature_runtime.cache_clear()
    from ml.classifier import load_portable_forest

    load_portable_forest.cache_clear()


def refine_defect_mask_for_classification(mask: np.ndarray | None) -> np.ndarray | None:
    """Filter out soft reflection halos and isolate the core localized defect."""
    if mask is None or not np.any(mask):
        return mask
    mask_u8 = mask.astype(np.uint8)
    num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if num_labels <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = float(np.max(areas))
    valid_comps = np.where(areas >= max_area * 0.25)[0] + 1
    clean_mask = np.isin(comp_labels, valid_comps).astype(np.uint8)
    return clean_mask.astype(bool)


def classify_defect_type(
    image_path: str | Path,
    classifier_model_path: str | Path,
    defect_mask: np.ndarray | None = None,
    cnn_classifier_path: str | Path | None = None,
    compact_classifier_path: str | Path | None = None,
    global_features: np.ndarray | None = None,
) -> dict:
    from ml.classifier import (
        HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
        extract_features,
        extract_global_texture_features,
        extract_handcrafted_roi_shape_features,
        extract_roi_pixel_texture_features,
        extract_roi_shape_texture_features,
        extract_roi_texture_features,
        load_portable_forest,
        predict_portable_forest,
    )

    if "capsule" in Path(classifier_model_path).parts:
        defect_mask = refine_defect_mask_for_classification(defect_mask)
    classifier_model_path = Path(classifier_model_path)
    cnn_onnx_path = Path(cnn_classifier_path) if cnn_classifier_path is not None else None
    cnn_metadata_path = cnn_onnx_path.with_suffix(".json") if cnn_onnx_path is not None else None
    classifier_fallback_reason = None
    if cnn_onnx_path is not None and cnn_metadata_path is not None and cnn_onnx_path.exists() and cnn_metadata_path.exists():
        try:
            return predict_portable_cnn_defect_type(
                image_path,
                cnn_onnx_path,
                cnn_metadata_path,
                defect_mask=defect_mask,
            )
        except Exception as exc:
            classifier_fallback_reason = f"Portable CNN could not be used: {exc}"
            logger.warning(
                "subtype_classifier_fallback cnn=%s fallback=%s reason=%s",
                cnn_onnx_path,
                compact_classifier_path or classifier_model_path,
                exc,
            )

    compact_path = Path(compact_classifier_path) if compact_classifier_path is not None else None
    if compact_path is not None and compact_path.exists():
        runtime = load_portable_forest(str(compact_path), compact_path.stat().st_mtime_ns)
        feature_mode = str(runtime["feature_mode"])
        if feature_mode != HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
            raise DefectClassifierError(f"Unsupported compact classifier feature mode: {feature_mode}")
        features = extract_handcrafted_roi_shape_features([image_path], masks=[defect_mask])
        prediction = predict_portable_forest(compact_path, features)
        probabilities = prediction["probabilities"][0]
        classes = prediction["classes"]
        label = str(prediction["labels"][0])
        return {
            "defect_type": label,
            "confidence": round(float(max(probabilities)), 4),
            "class_probabilities": {
                str(class_name): round(float(probability), 4)
                for class_name, probability in zip(classes, probabilities, strict=True)
            },
            "classifier_engine": "portable_forest",
            "classifier_fallback_used": classifier_fallback_reason is not None,
            "classifier_fallback_reason": classifier_fallback_reason,
            "confidence_calibrated": False,
        }

    bundle = load_classifier_runtime(str(classifier_model_path))
    feature_mode = bundle.get("feature_mode")
    if feature_mode == HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
        features = extract_handcrafted_roi_shape_features([image_path], masks=[defect_mask])
    else:
        feature_extractor, preprocess, device = shared_feature_runtime()
    if feature_mode == ROI_PIXEL_TEXTURE_FEATURE_MODE:
        features = extract_roi_pixel_texture_features(
            [image_path],
            masks=[defect_mask],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
            global_features=global_features,
        )
    elif feature_mode == ROI_SHAPE_TEXTURE_FEATURE_MODE:
        features = extract_roi_shape_texture_features(
            [image_path],
            masks=[defect_mask],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
            global_features=global_features,
        )
    elif feature_mode == ROI_TEXTURE_FEATURE_MODE:
        features = extract_roi_texture_features(
            [image_path],
            masks=[defect_mask],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
            global_features=global_features,
        )
    elif feature_mode == GLOBAL_TEXTURE_FEATURE_MODE:
        features = extract_global_texture_features(
            [image_path],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
            global_features=global_features,
        )
    elif feature_mode != HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
        features = global_features
        if features is None:
            features = extract_features(
                [image_path],
                feature_extractor=feature_extractor,
                preprocess=preprocess,
                device=device,
            )

    classifier = bundle["classifier"]
    label = str(classifier.predict(features)[0])
    classes = getattr(classifier, "classes_", bundle.get("labels", []))
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
    elif hasattr(classifier, "decision_function"):
        decision = np.asarray(classifier.decision_function(features))[0]
        if decision.ndim == 0:
            prob = 1.0 / (1.0 + np.exp(-decision))
            probabilities = np.array([1.0 - prob, prob], dtype=np.float32)
        else:
            exp_scores = np.exp(decision - np.max(decision))
            probabilities = exp_scores / max(float(exp_scores.sum()), 1e-6)
    else:
        probabilities = np.zeros(len(classes), dtype=np.float32)
        match_idx = list(classes).index(label) if label in list(classes) else 0
        probabilities[match_idx] = 1.0

    class_probabilities = {
        str(class_name): round(float(probability), 4)
        for class_name, probability in zip(classes, probabilities, strict=False)
    }
    return {
        "defect_type": label,
        "confidence": round(float(max(probabilities)), 4),
        "class_probabilities": class_probabilities,
        "classifier_engine": f"sklearn_{feature_mode or 'resnet18_features'}",
        "classifier_fallback_used": classifier_fallback_reason is not None,
        "classifier_fallback_reason": classifier_fallback_reason,
        "confidence_calibrated": False,
    }
