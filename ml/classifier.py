from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import cv2
import joblib
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

GLOBAL_FEATURE_MODE = "global"
GLOBAL_TEXTURE_FEATURE_MODE = "global_texture"
ROI_TEXTURE_FEATURE_MODE = "global_roi_texture"
ROI_SHAPE_TEXTURE_FEATURE_MODE = "global_roi_shape_texture"
ROI_PIXEL_TEXTURE_FEATURE_MODE = "global_roi_pixel_texture"
HANDCRAFTED_ROI_SHAPE_FEATURE_MODE = "handcrafted_roi_shape_texture"
PORTABLE_MODEL_CACHE_SIZE = max(1, min(int(os.getenv("CLASSIFIER_MODEL_CACHE_SIZE", "2")), 4))
FEATURE_EXTRACTOR_NAME = "resnet18_imagenet1k_v1"
GLOBAL_TEXTURE_EXTRACTOR_NAME = "resnet18_imagenet1k_v1_plus_texture"
ROI_TEXTURE_EXTRACTOR_NAME = "resnet18_imagenet1k_v1_plus_roi_texture_geometry"
ROI_SHAPE_TEXTURE_EXTRACTOR_NAME = "resnet18_imagenet1k_v1_plus_roi_texture_mask_shape"
ROI_PIXEL_TEXTURE_EXTRACTOR_NAME = "resnet18_imagenet1k_v1_plus_roi_texture_mask_shape_pixels"
HANDCRAFTED_ROI_SHAPE_EXTRACTOR_NAME = "opencv_texture_gradient_color_mask_shape"
MASK_SHAPE_FEATURE_LENGTH = 284
ROI_PIXEL_FEATURE_LENGTH = 3072
DEFAULT_RESNET_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "inference" / "resnet18-f37072fd.pth"
DEFAULT_RESNET_ONNX_PATH = Path(__file__).resolve().parents[1] / "models" / "shared" / "resnet18_features.onnx"
DEFAULT_RESNET_OPENVINO_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "shared" / "resnet18_features_fp16.xml"
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OpenCVResNet18FeatureExtractor:
    """Run the frozen ResNet18 embedding model without a PyTorch runtime."""

    def __init__(self, model_path: str | Path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.lock = Lock()

    def extract(self, images: list[Image.Image], preprocess, batch_size: int = 16) -> np.ndarray:
        features: list[np.ndarray] = []
        with self.lock:
            for start in range(0, len(images), batch_size):
                batch = np.stack([preprocess(image) for image in images[start : start + batch_size]])
                self.net.setInput(batch.astype(np.float32, copy=False))
                features.append(self.net.forward().reshape(len(batch), -1))
        return np.vstack(features).astype(np.float32, copy=False)


class OpenVINOResNet18FeatureExtractor:
    """Run the shared FP16 feature extractor with a bounded OpenVINO CPU runtime."""

    def __init__(self, model_path: str | Path):
        import openvino as ov

        core = ov.Core()
        self.model = core.compile_model(
            str(model_path),
            "CPU",
            {"INFERENCE_NUM_THREADS": "1", "NUM_STREAMS": "1", "PERFORMANCE_HINT": "LATENCY"},
        )
        self.input = self.model.input(0)
        self.output = self.model.output(0)
        self.lock = Lock()

    def extract(self, images: list[Image.Image], preprocess, batch_size: int = 16) -> np.ndarray:
        features: list[np.ndarray] = []
        with self.lock:
            for start in range(0, len(images), batch_size):
                batch = np.stack([preprocess(image) for image in images[start : start + batch_size]])
                result = self.model({self.input: batch.astype(np.float32, copy=False)})
                features.append(np.asarray(result[self.output]).reshape(len(batch), -1))
        return np.vstack(features).astype(np.float32, copy=False)


def imagenet_preprocess(image: Image.Image) -> np.ndarray:
    """Match torchvision's ResNet18 resize, center-crop, and normalization."""
    image = image.convert("RGB")
    width, height = image.size
    if width < height:
        resized_width, resized_height = 256, int(256 * height / width)
    else:
        resized_height, resized_width = 256, int(256 * width / height)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (resized_width - 224) // 2
    top = (resized_height - 224) // 2
    array = np.asarray(resized.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
    normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def build_opencv_resnet18_feature_extractor(model_path: str | Path = DEFAULT_RESNET_ONNX_PATH):
    path = Path(model_path)
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Portable ResNet18 ONNX model not found: {path}")
    return OpenCVResNet18FeatureExtractor(path), imagenet_preprocess, "cpu"


def build_openvino_resnet18_feature_extractor(model_path: str | Path = DEFAULT_RESNET_OPENVINO_PATH):
    path = Path(model_path)
    if not path.exists() or not path.with_suffix(".bin").exists():
        raise FileNotFoundError(f"OpenVINO FP16 ResNet18 model not found: {path}")
    return OpenVINOResNet18FeatureExtractor(path), imagenet_preprocess, "cpu"


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18_feature_extractor(device=None):
    """Build the shared frozen ImageNet ResNet18 feature extractor."""
    import torch
    from torchvision import models

    device = device or get_device()
    weights = models.ResNet18_Weights.DEFAULT
    has_bundled_weights = (
        DEFAULT_RESNET_WEIGHTS_PATH.exists() and DEFAULT_RESNET_WEIGHTS_PATH.stat().st_size > 1_000_000
    )
    if has_bundled_weights:
        model = models.resnet18(weights=None)
        state_dict = torch.load(DEFAULT_RESNET_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = models.resnet18(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval().to(device)
    return feature_extractor, weights.transforms(), device


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def extract_pil_features(
    images: list[Image.Image],
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device=None,
) -> np.ndarray:
    if feature_extractor is None or preprocess is None:
        if DEFAULT_RESNET_OPENVINO_PATH.exists() and DEFAULT_RESNET_OPENVINO_PATH.with_suffix(".bin").exists():
            feature_extractor, preprocess, device = build_openvino_resnet18_feature_extractor()
        elif DEFAULT_RESNET_ONNX_PATH.exists():
            feature_extractor, preprocess, device = build_opencv_resnet18_feature_extractor()
        else:
            feature_extractor, preprocess, device = build_resnet18_feature_extractor(device)
    if isinstance(feature_extractor, (OpenCVResNet18FeatureExtractor, OpenVINOResNet18FeatureExtractor)):
        return feature_extractor.extract(images, preprocess, batch_size=batch_size)

    import torch

    if device is None:
        device = get_device()

    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            tensors = [preprocess(image) for image in batch_images]
            embeddings = feature_extractor(torch.stack(tensors).to(device))
            features.append(embeddings.flatten(start_dim=1).cpu().numpy())
    return np.vstack(features)


def extract_features(
    image_paths: list[str | Path],
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device=None,
) -> np.ndarray:
    """Extract global ResNet18 embeddings for a list of image paths."""
    return extract_pil_features(
        [load_rgb_image(path) for path in image_paths],
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )


def handcrafted_array_features(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    orientation = cv2.phase(gradient_x, gradient_y, angleInDegrees=True) % 180.0

    features: list[float] = []
    for row in range(4):
        for column in range(4):
            row_slice = slice(row * 32, (row + 1) * 32)
            column_slice = slice(column * 32, (column + 1) * 32)
            histogram, _ = np.histogram(
                orientation[row_slice, column_slice].ravel(),
                bins=9,
                range=(0, 180),
                weights=magnitude[row_slice, column_slice].ravel(),
            )
            histogram = histogram.astype(np.float32)
            histogram /= max(float(np.linalg.norm(histogram)), 1e-9)
            features.extend(histogram.tolist())
            block = gray[row_slice, column_slice]
            features.extend([float(block.mean() / 255.0), float(block.std() / 255.0)])

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    for channel, bins, value_range in ((0, 18, (0, 180)), (1, 16, (0, 256)), (2, 16, (0, 256))):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], value_range).ravel()
        histogram /= max(float(histogram.sum()), 1.0)
        features.extend(histogram.tolist())

    center = gray[1:-1, 1:-1]
    lbp = np.zeros(center.shape, dtype=np.uint8)
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    for bit, neighbor in enumerate(neighbors):
        lbp |= (neighbor >= center).astype(np.uint8) << bit
    histogram, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    features.extend(histogram.tolist())

    return np.asarray(features, dtype=np.float32)


def handcrafted_image_features(image_path: str | Path) -> np.ndarray:
    """Extract compact gradient, intensity, texture, and color descriptors."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return handcrafted_array_features(image)


def load_mask(
    mask_path: str | Path | None = None, mask: np.ndarray | None = None, image_shape: tuple[int, int] | None = None
) -> np.ndarray | None:
    if mask is not None:
        mask_array = np.asarray(mask)
    elif mask_path:
        if not isinstance(mask_path, str | Path):
            return None
        if str(mask_path).strip().lower() in {"", "nan", "none"}:
            return None
        path = Path(mask_path)
        if not path.exists():
            return None
        mask_array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            return None
    else:
        return None

    mask_bool = mask_array.astype(bool)
    if image_shape and mask_bool.shape != image_shape:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask_bool


def defect_bbox(mask: np.ndarray, padding_ratio: float = 0.18) -> tuple[int, int, int, int] | None:
    if mask is None or not np.any(mask):
        return None
    height, width = mask.shape[:2]
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = int(round(max(x2 - x1, y2 - y1, min(height, width) * 0.08) * padding_ratio))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def crop_bgr_from_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    box = defect_bbox(mask) if mask is not None else None
    if box is None:
        return image
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def roi_pil_image(
    image_path: str | Path, mask_path: str | Path | None = None, mask: np.ndarray | None = None
) -> Image.Image:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    mask_bool = load_mask(mask_path, mask, image.shape[:2])
    roi = crop_bgr_from_mask(image, mask_bool)
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))


def handcrafted_roi_features(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    mask_bool = load_mask(mask_path, mask, image.shape[:2])
    return handcrafted_array_features(crop_bgr_from_mask(image, mask_bool))


def mask_geometry_features(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    mask_bool = load_mask(mask_path, mask, (height, width))
    if mask_bool is None or not np.any(mask_bool):
        return np.zeros(12, dtype=np.float32)

    area_ratio = float(mask_bool.mean())
    box = defect_bbox(mask_bool, padding_ratio=0.0)
    if box is None:
        return np.zeros(12, dtype=np.float32)
    x1, y1, x2, y2 = box
    bbox_width = (x2 - x1) / max(width, 1)
    bbox_height = (y2 - y1) / max(height, 1)
    center_x = ((x1 + x2) / 2) / max(width, 1)
    center_y = ((y1 + y2) / 2) / max(height, 1)
    aspect = bbox_width / max(bbox_height, 1e-6)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours) / max(height + width, 1)

    edge = np.zeros_like(mask_bool)
    border = max(1, int(round(min(height, width) * 0.12)))
    edge[:border, :] = True
    edge[-border:, :] = True
    edge[:, :border] = True
    edge[:, -border:] = True
    center_region = np.zeros_like(mask_bool)
    center_region[int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75)] = True
    defect_pixels = max(int(mask_bool.sum()), 1)

    return np.asarray(
        [
            area_ratio,
            bbox_width,
            bbox_height,
            center_x,
            center_y,
            min(aspect, 10.0) / 10.0,
            min(perimeter, 10.0) / 10.0,
            float(np.logical_and(mask_bool, edge).sum() / defect_pixels),
            float(np.logical_and(mask_bool, center_region).sum() / defect_pixels),
            float(mask_bool[: height // 2, :].sum() / defect_pixels),
            float(mask_bool[height // 2 :, :].sum() / defect_pixels),
            float(mask_bool[:, : width // 2].sum() / defect_pixels),
        ],
        dtype=np.float32,
    )


def mask_shape_features(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    mask_bool = load_mask(mask_path, mask, (height, width))
    if mask_bool is None or not np.any(mask_bool):
        return np.zeros(MASK_SHAPE_FEATURE_LENGTH, dtype=np.float32)

    mask_uint8 = mask_bool.astype(np.uint8)
    low_res = cv2.resize(mask_uint8.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    defect_area = max(float(mask_uint8.sum()), 1.0)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.asarray([], dtype=np.float32)
    largest_component_ratio = float(component_areas.max() / defect_area) if component_areas.size else 0.0
    normalized_component_count = min(max(component_count - 1, 0), 10) / 10.0

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extent = 0.0
    solidity = 0.0
    circularity = 0.0
    hu_values = np.zeros(7, dtype=np.float32)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        extent = contour_area / max(float(bbox_width * bbox_height), 1.0)
        hull = cv2.convexHull(contour)
        solidity = contour_area / max(float(cv2.contourArea(hull)), 1.0)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        circularity = min((4.0 * np.pi * contour_area) / (perimeter * perimeter), 1.0)
        moments = cv2.moments(contour)
        hu_raw = cv2.HuMoments(moments).ravel()
        hu_values = np.asarray(
            [-np.sign(value) * np.log10(abs(value) + 1e-12) / 20.0 for value in hu_raw], dtype=np.float32
        )

    horizontal_profile = cv2.resize(mask_uint8.astype(np.float32), (1, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    vertical_profile = cv2.resize(mask_uint8.astype(np.float32), (8, 1), interpolation=cv2.INTER_AREA).reshape(-1)
    extra = np.asarray(
        [
            normalized_component_count,
            largest_component_ratio,
            extent,
            solidity,
            circularity,
            *hu_values.tolist(),
            *horizontal_profile.tolist(),
            *vertical_profile.tolist(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([low_res.astype(np.float32), extra])


def roi_pixel_features(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    mask_bool = load_mask(mask_path, mask, image.shape[:2])
    roi = crop_bgr_from_mask(image, mask_bool)
    roi_resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    magnitude /= max(float(magnitude.max()), 1e-6)
    if mask_bool is None or not np.any(mask_bool):
        roi_mask = np.ones((32, 32), dtype=np.float32)
    else:
        cropped_mask = crop_bgr_from_mask(np.repeat(mask_bool[..., None].astype(np.uint8), 3, axis=2), mask_bool)
        roi_mask = cv2.resize(cropped_mask[:, :, 0].astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
        roi_mask = np.clip(roi_mask, 0, 1)
    return np.concatenate([gray.reshape(-1), magnitude.reshape(-1), roi_mask.reshape(-1)]).astype(np.float32)


def extract_global_texture_features(
    image_paths: list[str | Path],
    *,
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device=None,
    global_features: np.ndarray | None = None,
) -> np.ndarray:
    """Combine semantic ResNet context with local gradient and color cues."""
    if global_features is None:
        global_features = extract_features(
            image_paths,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    texture_features = np.vstack([handcrafted_image_features(path) for path in image_paths])
    return np.concatenate([global_features, texture_features], axis=1)


def extract_roi_texture_features(
    image_paths: list[str | Path],
    *,
    mask_paths: list[str | Path | None] | None = None,
    masks: list[np.ndarray | None] | None = None,
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device=None,
    global_features: np.ndarray | None = None,
) -> np.ndarray:
    """Combine whole-image context with defect-region features and mask geometry."""
    mask_paths = mask_paths or [None] * len(image_paths)
    masks = masks or [None] * len(image_paths)
    if global_features is None:
        global_features = extract_features(
            image_paths,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    roi_images = [
        roi_pil_image(path, mask_path=mask_path, mask=mask)
        for path, mask_path, mask in zip(image_paths, mask_paths, masks, strict=False)
    ]
    roi_global_features = extract_pil_features(
        roi_images,
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    global_texture = np.vstack([handcrafted_image_features(path) for path in image_paths])
    roi_texture = np.vstack(
        [
            handcrafted_roi_features(path, mask_path=mask_path, mask=mask)
            for path, mask_path, mask in zip(image_paths, mask_paths, masks, strict=False)
        ]
    )
    geometry = np.vstack(
        [
            mask_geometry_features(path, mask_path=mask_path, mask=mask)
            for path, mask_path, mask in zip(image_paths, mask_paths, masks, strict=False)
        ]
    )
    return np.concatenate([global_features, roi_global_features, global_texture, roi_texture, geometry], axis=1)


def extract_roi_shape_texture_features(
    image_paths: list[str | Path],
    *,
    mask_paths: list[str | Path | None] | None = None,
    masks: list[np.ndarray | None] | None = None,
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device=None,
    global_features: np.ndarray | None = None,
) -> np.ndarray:
    mask_paths = mask_paths or [None] * len(image_paths)
    masks = masks or [None] * len(image_paths)
    base_features = extract_roi_texture_features(
        image_paths,
        mask_paths=mask_paths,
        masks=masks,
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
        global_features=global_features,
    )
    shape_features = np.vstack(
        [
            mask_shape_features(path, mask_path=mask_path, mask=mask)
            for path, mask_path, mask in zip(image_paths, mask_paths, masks, strict=False)
        ]
    )
    return np.concatenate([base_features, shape_features], axis=1)


def extract_roi_pixel_texture_features(
    image_paths: list[str | Path],
    *,
    mask_paths: list[str | Path | None] | None = None,
    masks: list[np.ndarray | None] | None = None,
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device=None,
    global_features: np.ndarray | None = None,
) -> np.ndarray:
    mask_paths = mask_paths or [None] * len(image_paths)
    masks = masks or [None] * len(image_paths)
    base_features = extract_roi_shape_texture_features(
        image_paths,
        mask_paths=mask_paths,
        masks=masks,
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
        global_features=global_features,
    )
    pixel_features = np.vstack(
        [
            roi_pixel_features(path, mask_path=mask_path, mask=mask)
            for path, mask_path, mask in zip(image_paths, mask_paths, masks, strict=False)
        ]
    )
    return np.concatenate([base_features, pixel_features], axis=1)


def extract_handcrafted_roi_shape_features(
    image_paths: list[str | Path],
    *,
    mask_paths: list[str | Path | None] | None = None,
    masks: list[np.ndarray | None] | None = None,
) -> np.ndarray:
    """Extract portable global/ROI texture and mask-shape features without a CNN runtime."""
    mask_paths = mask_paths or [None] * len(image_paths)
    masks = masks or [None] * len(image_paths)
    rows = []
    for path, mask_path, mask in zip(image_paths, mask_paths, masks, strict=False):
        rows.append(
            np.concatenate(
                [
                    handcrafted_image_features(path),
                    handcrafted_roi_features(path, mask_path=mask_path, mask=mask),
                    mask_geometry_features(path, mask_path=mask_path, mask=mask),
                    mask_shape_features(path, mask_path=mask_path, mask=mask),
                ]
            )
        )
    return np.vstack(rows).astype(np.float32, copy=False)


def create_estimator(kind: str = "logistic", regularization: float = 1.0):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC

    if kind == "linear_svc":
        base = LinearSVC(C=regularization, random_state=42, max_iter=4000)
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    if kind == "ridge":
        return RidgeClassifier(alpha=regularization, random_state=42)
    if kind == "svc":
        return SVC(
            C=regularization,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    if kind == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(regularization),
            weights="distance",
        )
    if kind == "sgd_log_loss":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=regularization,
            l1_ratio=0.15,
            max_iter=2000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )
    return LogisticRegression(
        C=regularization,
        max_iter=4000,
        class_weight="balanced",
        random_state=42,
    )


def create_classifier(kind: str = "logistic", regularization: float = 1.0) -> Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = create_estimator(kind, regularization)
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def create_pca_classifier(kind: str = "logistic", regularization: float = 1.0) -> Pipeline:
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = create_estimator(kind, regularization)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, whiten=True, random_state=42)),
            ("classifier", estimator),
        ]
    )


def classifier_candidates(feature_count: int | None = None) -> dict[str, Pipeline]:
    if feature_count is not None and feature_count > 3000:
        return {
            "sgd_log_alpha0.0001": create_classifier("sgd_log_loss", 0.0001),
            "sgd_log_alpha0.001": create_classifier("sgd_log_loss", 0.001),
            "pca_logistic_c0.1": create_pca_classifier("logistic", 0.1),
            "pca_svc_rbf_c5": create_pca_classifier("svc", 5.0),
            "linear_svc_c0.01": create_classifier("linear_svc", 0.01),
            "linear_svc_c0.05": create_classifier("linear_svc", 0.05),
            "extra_trees": create_classifier("extra_trees"),
            "random_forest": create_classifier("random_forest"),
        }

    base_candidates = {
        "linear_svc_c0.01": create_classifier("linear_svc", 0.01),
        "linear_svc_c0.05": create_classifier("linear_svc", 0.05),
        "linear_svc_c0.1": create_classifier("linear_svc", 0.1),
        "logistic_c0.01": create_classifier("logistic", 0.01),
        "logistic_c0.1": create_classifier("logistic", 0.1),
        "logistic_c1": create_classifier("logistic", 1.0),
        "logistic_c10": create_classifier("logistic", 10.0),
        "lda_shrinkage": create_classifier("lda"),
        "extra_trees": create_classifier("extra_trees"),
        "random_forest": create_classifier("random_forest"),
        "gradient_boosting": create_classifier("gradient_boosting"),
    }
    return {
        **base_candidates,
        "svc_rbf_c1": create_classifier("svc", 1.0),
        "svc_rbf_c5": create_classifier("svc", 5.0),
        "svc_rbf_c20": create_classifier("svc", 20.0),
        "knn_3": create_classifier("knn", 3),
        "knn_5": create_classifier("knn", 5),
    }


def _cross_validated_candidate_scores(
    candidates: dict[str, Pipeline],
    features: np.ndarray,
    labels: np.ndarray,
    splitter,
) -> dict[str, dict[str, float]]:
    from sklearn.model_selection import cross_validate

    fold_workers = min(4, splitter.get_n_splits())
    scores_by_name: dict[str, dict[str, float]] = {}
    for name, candidate in candidates.items():
        scores = cross_validate(
            candidate,
            features,
            labels,
            cv=splitter,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=fold_workers,
            pre_dispatch=fold_workers,
        )
        scores_by_name[name] = {
            "mean_accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
            "std_accuracy": round(float(np.std(scores["test_accuracy"])), 4),
            "mean_macro_f1": round(float(np.mean(scores["test_macro_f1"])), 4),
            "std_macro_f1": round(float(np.std(scores["test_macro_f1"])), 4),
        }
    return scores_by_name


def _best_candidate_name(candidate_scores: dict[str, dict[str, float]]) -> str:
    return max(
        candidate_scores,
        key=lambda name: (
            candidate_scores[name]["mean_macro_f1"],
            candidate_scores[name]["mean_accuracy"],
            -candidate_scores[name]["std_macro_f1"],
        ),
    )


def select_classifier(features: np.ndarray, labels: np.ndarray, random_state: int = 42) -> tuple[str, Pipeline, dict]:
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    class_counts = pd.Series(labels).value_counts()
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        raise ValueError("At least two labelled images per defect subtype are required.")
    candidates = classifier_candidates(features.shape[1])
    outer_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_predictions = np.empty(len(labels), dtype=object)
    outer_selections: list[str] = []

    # Model choice happens only inside each outer training fold. The held-out
    # outer fold therefore remains unseen until its final prediction.
    for outer_fold, (train_index, test_index) in enumerate(outer_splitter.split(features, labels), start=1):
        train_labels = labels[train_index]
        inner_minimum = int(pd.Series(train_labels).value_counts().min())
        inner_folds = min(4, inner_minimum)
        if inner_folds < 2:
            fold_name = "logistic_c1"
        else:
            inner_splitter = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=random_state + outer_fold,
            )
            inner_scores = _cross_validated_candidate_scores(
                candidates,
                features[train_index],
                train_labels,
                inner_splitter,
            )
            fold_name = _best_candidate_name(inner_scores)
        fold_classifier = clone(candidates[fold_name])
        fold_classifier.fit(features[train_index], train_labels)
        oof_predictions[test_index] = fold_classifier.predict(features[test_index])
        outer_selections.append(fold_name)

    # Select the production estimator on all available labelled data after
    # nested evaluation has finished, then fit it once for runtime use.
    production_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state + 101)
    candidate_scores = _cross_validated_candidate_scores(candidates, features, labels, production_splitter)
    best_name = _best_candidate_name(candidate_scores)
    selected = candidates[best_name]
    selection_frequency = {
        name: outer_selections.count(name) for name in sorted(set(outer_selections))
    }
    return (
        best_name,
        selected,
        {
            "folds": folds,
            "candidate_scores": candidate_scores,
            "outer_selection_frequency": selection_frequency,
            "oof_predictions": oof_predictions,
        },
    )


def _extract_training_features(
    data: pd.DataFrame,
    feature_mode: str,
    batch_size: int,
    feature_extractor,
    preprocess,
    device,
) -> np.ndarray:
    paths = data["image_path"].tolist()
    masks = data["mask"].tolist() if "mask" in data.columns else None
    if feature_mode == HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
        mask_paths = data["mask_path"].tolist() if "mask_path" in data.columns else None
        return extract_handcrafted_roi_shape_features(paths, mask_paths=mask_paths, masks=masks)
    if feature_mode == GLOBAL_TEXTURE_FEATURE_MODE:
        return extract_global_texture_features(
            paths,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    if feature_mode == ROI_TEXTURE_FEATURE_MODE:
        mask_paths = data["mask_path"].tolist() if "mask_path" in data.columns else None
        return extract_roi_texture_features(
            paths,
            mask_paths=mask_paths,
            masks=masks,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    if feature_mode == ROI_SHAPE_TEXTURE_FEATURE_MODE:
        mask_paths = data["mask_path"].tolist() if "mask_path" in data.columns else None
        return extract_roi_shape_texture_features(
            paths,
            mask_paths=mask_paths,
            masks=masks,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    if feature_mode == ROI_PIXEL_TEXTURE_FEATURE_MODE:
        mask_paths = data["mask_path"].tolist() if "mask_path" in data.columns else None
        return extract_roi_pixel_texture_features(
            paths,
            mask_paths=mask_paths,
            masks=masks,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    return extract_features(
        paths,
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )


def train_defect_classifier(
    dataset_df: pd.DataFrame,
    output_path: str | Path,
    test_size: float = 0.3,
    random_state: int = 42,
    batch_size: int = 16,
    label_order: list[str] | None = None,
    dataset_context: dict | None = None,
    *,
    defect_only: bool = False,
    feature_mode: str = GLOBAL_FEATURE_MODE,
    cross_validate_model: bool = False,
) -> dict:
    """Train a compact category-specific classifier from labelled images."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    data = dataset_df.copy()
    if defect_only:
        data = data[data["label"] != "good"].copy()
    labels = label_order or sorted(data["label"].unique().tolist())
    labels = [label for label in labels if label in set(data["label"])]
    data = data[data["label"].isin(labels)].copy()
    if data.empty:
        raise ValueError("Dataset is empty. Cannot train classifier.")

    if feature_mode == HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
        feature_extractor = preprocess = device = None
    else:
        feature_extractor, preprocess, device = build_resnet18_feature_extractor()
    features = _extract_training_features(
        data,
        feature_mode,
        batch_size,
        feature_extractor,
        preprocess,
        device,
    )
    y = np.asarray(data["label"].tolist())
    extractor_names = {
        GLOBAL_FEATURE_MODE: FEATURE_EXTRACTOR_NAME,
        GLOBAL_TEXTURE_FEATURE_MODE: GLOBAL_TEXTURE_EXTRACTOR_NAME,
        ROI_TEXTURE_FEATURE_MODE: ROI_TEXTURE_EXTRACTOR_NAME,
        ROI_SHAPE_TEXTURE_FEATURE_MODE: ROI_SHAPE_TEXTURE_EXTRACTOR_NAME,
        ROI_PIXEL_TEXTURE_FEATURE_MODE: ROI_PIXEL_TEXTURE_EXTRACTOR_NAME,
        HANDCRAFTED_ROI_SHAPE_FEATURE_MODE: HANDCRAFTED_ROI_SHAPE_EXTRACTOR_NAME,
    }
    extractor_name = extractor_names.get(feature_mode, FEATURE_EXTRACTOR_NAME)
    context = dataset_context or {}

    if len(labels) == 1:
        classifier = DummyClassifier(strategy="constant", constant=labels[0])
        classifier.fit(features, y)
        y_eval = y
        y_pred = classifier.predict(features)
        train_df = data
        eval_df = data
        evaluation = {
            "protocol": "single known defect subtype; fitted on all labelled defect images",
            "selected_classifier": "constant",
            "folds": 0,
            "candidate_scores": {},
        }
    elif cross_validate_model:
        selected_name, classifier, selection = select_classifier(features, y, random_state=random_state)
        y_eval = y
        y_pred = selection.pop("oof_predictions")
        classifier.fit(features, y)
        train_df = data
        eval_df = data
        evaluation = {
            "protocol": (
                "nested stratified cross-validation; model selection occurs inside each outer training fold, "
                "and the production classifier is fitted on all labelled defect images only after evaluation"
            ),
            "selected_classifier": selected_name,
            **selection,
        }
    else:
        train_index, eval_index = train_test_split(
            np.arange(len(data)),
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        classifier = create_classifier()
        classifier.fit(features[train_index], y[train_index])
        y_eval = y[eval_index]
        y_pred = classifier.predict(features[eval_index])
        train_df = data.iloc[train_index]
        eval_df = data.iloc[eval_index]
        classifier.fit(features, y)
        evaluation = {
            "protocol": (
                "stratified holdout evaluation; production classifier refitted on all labelled defect images "
                "after evaluation"
            ),
            "selected_classifier": "logistic_c1",
            "folds": 0,
            "candidate_scores": {},
        }

    metrics = {
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_eval, y_pred, average="macro")), 4),
        "classification_report": classification_report(
            y_eval,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=labels).tolist(),
        "labels": labels,
        "train_size": int(len(train_df)),
        "eval_size": int(len(eval_df)),
        "feature_extractor": extractor_name,
        "feature_mode": feature_mode,
        "defect_only": defect_only,
        "evaluation": evaluation,
        "dataset_context": context,
    }
    bundle = {
        "classifier": classifier,
        "labels": labels,
        "metrics": metrics,
        "feature_extractor": extractor_name,
        "feature_mode": feature_mode,
        "defect_only": defect_only,
        "image_size": [224, 224],
        "dataset_context": context,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return {
        "bundle": bundle,
        "train_df": train_df,
        "eval_df": eval_df,
        "y_eval": list(y_eval),
        "y_pred": list(y_pred),
        "metrics": metrics,
        "output_path": output_path,
    }


def load_classifier_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def export_portable_forest(
    classifier,
    output_path: str | Path,
    *,
    feature_mode: str,
    decision_threshold: float = 0.5,
) -> Path:
    """Export a fitted sklearn forest or logistic model for NumPy-only inference."""
    estimator = classifier
    scaler_mean = np.empty(0, dtype=np.float32)
    scaler_scale = np.empty(0, dtype=np.float32)
    if hasattr(classifier, "named_steps"):
        estimator = classifier.steps[-1][1]
        scaler = classifier.named_steps.get("scaler")
        if scaler is None and classifier.steps and hasattr(classifier.steps[0][1], "mean_"):
            scaler = classifier.steps[0][1]
        if scaler is not None:
            scaler_mean = np.asarray(scaler.mean_, dtype=np.float32)
            scaler_scale = np.asarray(scaler.scale_, dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        np.savez_compressed(
            output_path,
            model_type=np.asarray("logistic"),
            coefficients=np.asarray(estimator.coef_, dtype=np.float32),
            intercept=np.asarray(estimator.intercept_, dtype=np.float32),
            classes=np.asarray(estimator.classes_),
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_mode=np.asarray(feature_mode),
            decision_threshold=np.asarray(decision_threshold, dtype=np.float32),
        )
        return output_path
    if not hasattr(estimator, "estimators_"):
        raise ValueError("Portable export requires a fitted tree ensemble or logistic model")

    offsets = [0]
    children_left = []
    children_right = []
    split_features = []
    thresholds = []
    values = []
    for tree_estimator in estimator.estimators_:
        tree = tree_estimator.tree_
        offset = offsets[-1]
        left = tree.children_left.astype(np.int32)
        right = tree.children_right.astype(np.int32)
        children_left.append(np.where(left >= 0, left + offset, left))
        children_right.append(np.where(right >= 0, right + offset, right))
        split_features.append(tree.feature.astype(np.int32))
        thresholds.append(tree.threshold.astype(np.float32))
        values.append(tree.value[:, 0, :].astype(np.float32))
        offsets.append(offsets[-1] + tree.node_count)

    np.savez_compressed(
        output_path,
        model_type=np.asarray("forest"),
        tree_offsets=np.asarray(offsets, dtype=np.int32),
        children_left=np.concatenate(children_left),
        children_right=np.concatenate(children_right),
        split_features=np.concatenate(split_features),
        thresholds=np.concatenate(thresholds),
        values=np.concatenate(values),
        classes=np.asarray(estimator.classes_),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_mode=np.asarray(feature_mode),
        decision_threshold=np.asarray(decision_threshold, dtype=np.float32),
    )
    return output_path


@lru_cache(maxsize=PORTABLE_MODEL_CACHE_SIZE)
def load_portable_forest(path_value: str, modified_ns: int) -> dict:
    del modified_ns
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Portable classifier not found: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        return {name: artifact[name].copy() for name in artifact.files}


def predict_portable_forest(path_value: str | Path, features: np.ndarray) -> dict:
    path = Path(path_value)
    runtime = load_portable_forest(str(path), path.stat().st_mtime_ns)
    matrix = np.atleast_2d(np.asarray(features, dtype=np.float32))
    if runtime["scaler_mean"].size:
        matrix = (matrix - runtime["scaler_mean"]) / np.maximum(runtime["scaler_scale"], 1e-12)

    model_type = str(runtime.get("model_type", np.asarray("forest")))
    if model_type == "logistic":
        logits = matrix @ runtime["coefficients"].T + runtime["intercept"]
        if logits.shape[1] == 1 and len(runtime["classes"]) == 2:
            positive = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -50.0, 50.0)))
            probabilities = np.column_stack([1.0 - positive, positive])
        else:
            shifted = logits - logits.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    else:
        offsets = runtime["tree_offsets"]
        probabilities = np.zeros((len(matrix), len(runtime["classes"])), dtype=np.float64)
        for row_index, row in enumerate(matrix):
            for tree_index in range(len(offsets) - 1):
                node = int(offsets[tree_index])
                tree_end = int(offsets[tree_index + 1])
                while runtime["children_left"][node] >= 0:
                    feature = int(runtime["split_features"][node])
                    node = int(
                        runtime["children_left"][node]
                        if row[feature] <= runtime["thresholds"][node]
                        else runtime["children_right"][node]
                    )
                    if node < offsets[tree_index] or node >= tree_end:
                        raise ValueError("Portable forest contains an invalid node reference")
                leaf_values = runtime["values"][node]
                probabilities[row_index] += leaf_values / max(float(leaf_values.sum()), 1.0)
        probabilities /= max(len(offsets) - 1, 1)
    labels = runtime["classes"][np.argmax(probabilities, axis=1)]
    return {
        "labels": labels,
        "probabilities": probabilities,
        "classes": runtime["classes"],
        "feature_mode": str(runtime["feature_mode"]),
        "decision_threshold": float(runtime["decision_threshold"]),
    }
