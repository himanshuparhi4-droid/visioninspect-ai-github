"""Category-specific model locations for the multi-product inspection system."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ml.config import MODELS_DIR

SUPPORTED_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

CATEGORY_DEFECT_LABELS = {
    "bottle": ("broken_large", "broken_small", "contamination"),
    "cable": (
        "bent_wire",
        "cable_swap",
        "combined",
        "cut_inner_insulation",
        "cut_outer_insulation",
        "missing_cable",
        "missing_wire",
        "poke_insulation",
    ),
    "capsule": ("crack", "faulty_imprint", "poke", "scratch", "squeeze"),
    "carpet": ("color", "cut", "hole", "metal_contamination", "thread"),
    "grid": ("bent", "broken", "glue", "metal_contamination", "thread"),
    "hazelnut": ("crack", "cut", "hole", "print"),
    "leather": ("color", "cut", "fold", "glue", "poke"),
    "metal_nut": ("bent", "color", "flip", "scratch"),
    "pill": ("color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"),
    "screw": ("manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"),
    "tile": ("crack", "glue_strip", "gray_stroke", "oil", "rough"),
    "toothbrush": ("defective",),
    "transistor": ("bent_lead", "cut_lead", "damaged_case", "misplaced"),
    "wood": ("color", "combined", "hole", "liquid", "scratch"),
    "zipper": (
        "broken_teeth",
        "combined",
        "fabric_border",
        "fabric_interior",
        "rough",
        "split_teeth",
        "squeezed_teeth",
    ),
}

LOW_MEMORY_CLASSIFIER_ENGINES = frozenset(
    {
        "fine_tuned_resnet18_onnx",
        "portable_forest",
    }
)


def is_valid_checkpoint(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1_000_000


@lru_cache(maxsize=128)
def _artifact_descriptor(path_value: str, size: int, modified_ns: int) -> dict:
    path = Path(path_value)
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "file": path.name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def artifact_descriptor(path: Path) -> dict | None:
    """Return stable integrity metadata without exposing local absolute paths."""
    if not path.exists() or not path.is_file():
        return None
    stat = path.stat()
    return _artifact_descriptor(str(path), stat.st_size, stat.st_mtime_ns)


class CategoryModelError(ValueError):
    pass


@dataclass(frozen=True)
class CategoryModelSpec:
    category: str
    model_kind: str
    checkpoint_path: Path
    classifier_path: Path
    baseline_profile_path: Path
    metadata_path: Path
    cnn_classifier_path: Path | None = None
    openvino_path: Path | None = None
    openvino_calibrator_path: Path | None = None
    portable_detector_calibrator_path: Path | None = None
    compact_classifier_path: Path | None = None
    padim_score_threshold: float = 0.5
    baseline_score_threshold: float = 1.34
    baseline_residual_threshold: float = 1.34
    subtype_confidence_threshold: float = 0.55
    input_size: int = 256

    @property
    def has_advanced_model(self) -> bool:
        checkpoint_ready = is_valid_checkpoint(self.checkpoint_path)
        openvino_ready = False
        if self.openvino_path is not None and self.openvino_path.exists():
            openvino_ready = self.openvino_path.with_suffix(".bin").exists()
        return checkpoint_ready or openvino_ready

    @property
    def is_runnable(self) -> bool:
        return self.baseline_profile_path.exists() and self.metadata_path.exists()

    @property
    def is_fully_ready(self) -> bool:
        return self.is_runnable and self.classifier_path.exists()

    @property
    def is_trained(self) -> bool:
        """Backward-compatible name for an available advanced anomaly model."""
        return self.has_advanced_model


def normalize_category(value: str | None) -> str:
    normalized = (value or "bottle").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in SUPPORTED_CATEGORIES:
        supported = ", ".join(SUPPORTED_CATEGORIES)
        raise CategoryModelError(f"Unsupported product category '{value}'. Supported categories: {supported}.")
    return normalized


def registry_file() -> Path:
    return MODELS_DIR / "category_model_registry.json"


def _default_spec(category: str) -> CategoryModelSpec:
    category_dir = MODELS_DIR / "categories" / category
    return CategoryModelSpec(
        category=category,
        model_kind="padim",
        checkpoint_path=category_dir / "padim_v1.ckpt",
        classifier_path=category_dir / "defect_classifier.pkl",
        baseline_profile_path=category_dir / "normal_profile.npz",
        metadata_path=category_dir / "model_metadata.json",
        cnn_classifier_path=category_dir / "cnn_defect_classifier.onnx",
        openvino_path=MODELS_DIR / "exported" / category / "weights" / "openvino" / "model.xml",
        openvino_calibrator_path=category_dir / "openvino_calibrator.npz",
        portable_detector_calibrator_path=category_dir / "portable_detector_calibrator.npz",
        compact_classifier_path=category_dir / "defect_classifier_runtime.npz",
    )


@lru_cache(maxsize=4)
def _read_registry(path_value: str, modified_ns: int) -> dict:
    path = Path(path_value)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _registry_overrides() -> dict:
    path = registry_file()
    modified_ns = path.stat().st_mtime_ns if path.exists() else 0
    return _read_registry(str(path), modified_ns)


def category_model_spec(category: str | None) -> CategoryModelSpec:
    normalized = normalize_category(category)
    default = _default_spec(normalized)
    override = _registry_overrides().get(normalized, {})
    if not isinstance(override, dict):
        override = {}

    def required_path_value(name: str, fallback: Path) -> Path:
        value = override.get(name)
        return MODELS_DIR.parent / value if value else fallback

    def optional_path_value(name: str, fallback: Path | None) -> Path | None:
        if name in override and override[name] is None:
            return None
        value = override.get(name)
        return MODELS_DIR.parent / value if value else fallback

    return CategoryModelSpec(
        category=normalized,
        model_kind=str(override.get("model_kind", default.model_kind)).lower(),
        checkpoint_path=required_path_value("checkpoint_path", default.checkpoint_path),
        classifier_path=required_path_value("classifier_path", default.classifier_path),
        baseline_profile_path=required_path_value("baseline_profile_path", default.baseline_profile_path),
        metadata_path=required_path_value("metadata_path", default.metadata_path),
        cnn_classifier_path=optional_path_value("cnn_classifier_path", default.cnn_classifier_path),
        openvino_path=optional_path_value("openvino_path", default.openvino_path),
        openvino_calibrator_path=optional_path_value(
            "openvino_calibrator_path", default.openvino_calibrator_path
        ),
        portable_detector_calibrator_path=optional_path_value(
            "portable_detector_calibrator_path", default.portable_detector_calibrator_path
        ),
        compact_classifier_path=optional_path_value("compact_classifier_path", default.compact_classifier_path),
        padim_score_threshold=float(override.get("padim_score_threshold", default.padim_score_threshold)),
        baseline_score_threshold=float(override.get("baseline_score_threshold", default.baseline_score_threshold)),
        baseline_residual_threshold=float(
            override.get("baseline_residual_threshold", default.baseline_residual_threshold)
        ),
        subtype_confidence_threshold=float(
            override.get("subtype_confidence_threshold", default.subtype_confidence_threshold)
        ),
        input_size=int(override.get("input_size", default.input_size)),
    )


def _json_artifact_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8-sig")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def classifier_runtime_status(spec: CategoryModelSpec) -> dict:
    """Describe the first usable subtype runtime in production priority order."""
    if spec.cnn_classifier_path is not None:
        metadata_path = spec.cnn_classifier_path.with_suffix(".json")
        if spec.cnn_classifier_path.exists() and _json_artifact_is_valid(metadata_path):
            return {
                "engine": "fine_tuned_resnet18_onnx",
                "available": True,
                "artifact": artifact_descriptor(spec.cnn_classifier_path),
            }
    if spec.compact_classifier_path is not None and spec.compact_classifier_path.exists():
        return {
            "engine": "portable_forest",
            "available": True,
            "artifact": artifact_descriptor(spec.compact_classifier_path),
        }
    if spec.classifier_path.exists():
        return {
            "engine": "sklearn_feature_classifier",
            "available": True,
            "artifact": artifact_descriptor(spec.classifier_path),
        }
    return {"engine": "unavailable", "available": False, "artifact": None}


def openvino_runtime_is_memory_safe(spec: CategoryModelSpec, classifier_engine: str) -> bool:
    """Return whether the detector/classifier pair fits the constrained deployment profile."""
    return spec.model_kind == "padim" or classifier_engine in LOW_MEMORY_CLASSIFIER_ENGINES


def category_model_statuses(
    advanced_enabled: bool = False,
    openvino_enabled: bool = False,
    resource_constrained: bool = False,
) -> list[dict]:
    statuses = []
    for category in SUPPORTED_CATEGORIES:
        spec = category_model_spec(category)
        classifier = classifier_runtime_status(spec)
        openvino_available = bool(
            spec.openvino_path is not None
            and spec.openvino_path.exists()
            and spec.openvino_path.with_suffix(".bin").exists()
        )
        openvino_memory_safe = openvino_runtime_is_memory_safe(spec, classifier["engine"])
        openvino_requested = openvino_enabled and openvino_available
        openvino_ready = openvino_requested and (
            not resource_constrained or openvino_memory_safe
        )
        native_advanced_ready = (
            advanced_enabled
            and not resource_constrained
            and is_valid_checkpoint(spec.checkpoint_path)
        )
        active_engine = (
            f"{spec.model_kind}_openvino"
            if openvino_ready
            else spec.model_kind
            if native_advanced_ready
            else "portable_baseline"
        )
        statuses.append(
            {
                "category": category,
                "available": spec.is_runnable,
                "runnable": spec.is_runnable,
                "fully_ready": spec.is_runnable and classifier["available"],
                "trained": spec.has_advanced_model,
                "advanced_model_available": spec.has_advanced_model,
                "openvino_available": openvino_available,
                "openvino_memory_safe": openvino_memory_safe,
                "openvino_deferred_for_memory": bool(
                    resource_constrained and openvino_requested and not openvino_memory_safe
                ),
                "classification_trained": classifier["available"],
                "portable_cnn_available": classifier["engine"] == "fine_tuned_resnet18_onnx",
                "model_kind": spec.model_kind,
                "active_engine": active_engine,
                "deployment_tier": "advanced" if active_engine != "portable_baseline" else "portable",
                "classifier_engine": classifier["engine"],
                "subtype_labels": list(CATEGORY_DEFECT_LABELS[category]),
                "subtype_count": len(CATEGORY_DEFECT_LABELS[category]),
                "subtype_confidence_threshold": spec.subtype_confidence_threshold,
                "input_size": spec.input_size,
                "decision_threshold": spec.padim_score_threshold,
                "baseline_score_threshold": spec.baseline_score_threshold,
                "artifacts": {
                    "profile": artifact_descriptor(spec.baseline_profile_path),
                    "classifier": artifact_descriptor(spec.classifier_path),
                    "active_classifier": classifier["artifact"],
                    "cnn_classifier": artifact_descriptor(spec.cnn_classifier_path)
                    if spec.cnn_classifier_path
                    else None,
                    "openvino_calibrator": artifact_descriptor(spec.openvino_calibrator_path)
                    if spec.openvino_calibrator_path
                    else None,
                    "portable_detector_calibrator": artifact_descriptor(spec.portable_detector_calibrator_path)
                    if spec.portable_detector_calibrator_path
                    else None,
                    "compact_classifier": artifact_descriptor(spec.compact_classifier_path)
                    if spec.compact_classifier_path
                    else None,
                    "metadata": artifact_descriptor(spec.metadata_path),
                },
            }
        )
    return statuses
