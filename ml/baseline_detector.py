from pathlib import Path

import cv2
import numpy as np

DEFAULT_VARIABILITY_FLOOR = 15.0
DEFAULT_BASELINE_THRESHOLD = 1.34


def load_image_bgr(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_gray(image_bgr: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced.astype(np.float32)


def build_reference_image(train_image_paths: list[str | Path], size: tuple[int, int] = (256, 256)) -> np.ndarray:
    return build_reference_profile(train_image_paths, size=size)["mean"]


def foreground_mask(reference_image: np.ndarray) -> np.ndarray:
    """Find stable product pixels and exclude the bright background."""
    reference_uint8 = np.clip(reference_image, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(reference_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(bool)


def build_reference_profile(train_image_paths: list[str | Path], size: tuple[int, int] = (256, 256)) -> dict:
    """Create a normal-image profile used by the lightweight fallback detector."""
    if not train_image_paths:
        raise ValueError("No training images provided.")

    processed_images = [preprocess_gray(load_image_bgr(image_path), size=size) for image_path in train_image_paths]
    image_stack = np.stack(processed_images).astype(np.float32)
    mean_image = np.mean(image_stack, axis=0).astype(np.float32)

    return {
        "mean": mean_image,
        "std": np.std(image_stack, axis=0).astype(np.float32),
        "foreground_mask": foreground_mask(mean_image),
    }


def save_reference_profile(profile: dict, destination: str | Path) -> None:
    """Persist only the compact normal-profile arrays required at runtime."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": np.asarray(profile["mean"], dtype=np.float32),
        "std": np.asarray(profile["std"], dtype=np.float32),
        "foreground_mask": np.asarray(profile["foreground_mask"], dtype=bool),
    }
    if "embedding_bank" in profile:
        payload["embedding_bank"] = np.asarray(profile["embedding_bank"], dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_reference_profile(profile_path: str | Path) -> dict:
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline profile not found: {path}")

    with np.load(path) as profile_file:
        profile = {
            "mean": profile_file["mean"].astype(np.float32),
            "std": profile_file["std"].astype(np.float32),
            "foreground_mask": profile_file["foreground_mask"].astype(bool),
        }
        if "embedding_bank" in profile_file.files:
            profile["embedding_bank"] = profile_file["embedding_bank"].astype(np.float32)
        return profile


def normalize_embeddings(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)


def build_embedding_bank(
    train_image_paths: list[str | Path],
    *,
    batch_size: int = 32,
    maximum_images: int = 256,
) -> np.ndarray:
    """Build a compact memory bank of normal ResNet18 image embeddings."""
    if not train_image_paths:
        raise ValueError("No normal training images provided.")
    selected = list(train_image_paths)
    if len(selected) > maximum_images:
        indices = np.linspace(0, len(selected) - 1, maximum_images, dtype=int)
        selected = [selected[index] for index in indices]

    from ml.classifier import extract_features
    from ml.defect_classifier import shared_feature_runtime

    feature_extractor, preprocess, device = shared_feature_runtime()
    features = extract_features(
        selected,
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    return normalize_embeddings(features)


def embedding_anomaly_scores(features: np.ndarray, embedding_bank: np.ndarray) -> np.ndarray:
    """Return nearest-normal cosine distance for each image embedding."""
    normalized_features = normalize_embeddings(features)
    normalized_bank = normalize_embeddings(embedding_bank)
    return 1.0 - np.max(normalized_features @ normalized_bank.T, axis=1)


def embedding_anomaly_score(
    image_path: str | Path,
    embedding_bank: np.ndarray,
) -> float:
    score, _features = embedding_anomaly_evidence(image_path, embedding_bank)
    return score


def embedding_anomaly_evidence(
    image_path: str | Path,
    embedding_bank: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return anomaly score and reusable raw image features."""
    from ml.classifier import extract_features
    from ml.defect_classifier import shared_feature_runtime

    feature_extractor, preprocess, device = shared_feature_runtime()
    features = extract_features(
        [image_path],
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    return float(embedding_anomaly_scores(features, embedding_bank)[0]), features


def anomaly_map(image_bgr: np.ndarray, reference_image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    processed = preprocess_gray(image_bgr, size=size)
    diff = cv2.absdiff(processed, reference_image)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def normalized_anomaly_map(
    image_bgr: np.ndarray,
    reference_profile: dict,
    size: tuple[int, int] = (256, 256),
    variability_floor: float = DEFAULT_VARIABILITY_FLOOR,
) -> np.ndarray:
    """Score residuals relative to normal variation after lighting correction."""
    processed = preprocess_gray(image_bgr, size=size)
    reference = np.asarray(reference_profile["mean"], dtype=np.float32)
    variability = np.asarray(reference_profile["std"], dtype=np.float32)
    mask = np.asarray(reference_profile["foreground_mask"], dtype=bool)

    if processed.shape != reference.shape or reference.shape != variability.shape or mask.shape != reference.shape:
        raise ValueError("Baseline profile dimensions do not match the configured image size.")
    if not np.any(mask):
        raise ValueError("Baseline profile foreground mask is empty.")

    brightness_offset = float(np.median((processed - reference)[mask]))
    residual = np.abs(processed - brightness_offset - reference) / (variability + variability_floor)
    residual = cv2.GaussianBlur(residual.astype(np.float32), (5, 5), 0)
    residual[~mask] = 0.0
    return residual.astype(np.float32)


def anomaly_score(diff_map: np.ndarray, percentile: float = 99.0, mask: np.ndarray | None = None) -> float:
    values = diff_map[np.asarray(mask, dtype=bool)] if mask is not None else diff_map.reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def anomaly_mask(
    diff_map: np.ndarray,
    score_threshold: float,
    *,
    pixel_multiplier: float = 1.0,
) -> np.ndarray:
    """Remove low-level residual noise before geometry and heatmap reporting."""
    pixel_threshold = max(0.1, float(score_threshold) * pixel_multiplier)
    candidate = (diff_map > pixel_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    cleaned = np.zeros_like(candidate, dtype=bool)
    for label_index in range(1, count):
        if stats[label_index, cv2.CC_STAT_AREA] >= 4:
            cleaned[labels == label_index] = True
    return cleaned


def threshold_from_scores(scores: list[float], percentile: float = 99.0) -> float:
    if not scores:
        raise ValueError("No scores provided for threshold calculation.")
    return float(np.percentile(scores, percentile))


def predict_from_score(score: float, threshold: float) -> int:
    return int(score > threshold)


def evaluate_binary_predictions(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": matrix.tolist(),
    }


def heatmap_overlay(
    image_bgr: np.ndarray,
    diff_map: np.ndarray,
    alpha: float = 0.45,
    binary_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay anomaly color only where a localized defect survives filtering."""
    image_resized = cv2.resize(image_bgr, (diff_map.shape[1], diff_map.shape[0]), interpolation=cv2.INTER_AREA)
    if binary_mask is not None:
        mask = np.asarray(binary_mask, dtype=bool)
        if mask.shape != diff_map.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (diff_map.shape[1], diff_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not np.any(mask):
            return image_resized
        values = diff_map[mask]
    else:
        mask = np.ones(diff_map.shape, dtype=bool)
        values = diff_map.reshape(-1)

    lower, upper = np.percentile(values, [5, 99])
    if upper <= lower:
        return image_resized
    normalized = np.clip((diff_map - lower) / (upper - lower), 0, 1)
    normalized = (normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_resized, 1 - alpha, heatmap, alpha, 0)
    overlay = np.where(mask[..., None], blended, image_resized)
    return overlay
