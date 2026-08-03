from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def largest_component_mask(mask: np.ndarray, *, min_area_ratio: float = 0.002) -> np.ndarray:
    candidate = np.asarray(mask, dtype=np.uint8)
    if candidate.ndim == 3:
        candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    candidate = (candidate > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return candidate.astype(bool)

    image_area = candidate.shape[0] * candidate.shape[1]
    best_label = 0
    best_area = 0
    for label_index in range(1, count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label_index
    if best_area < image_area * min_area_ratio:
        return candidate.astype(bool)
    return (labels == best_label).astype(bool)


def foreground_mask_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Estimate object pixels for bright-background industrial product images."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_object = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_object = cv2.morphologyEx(dark_object, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark_object = cv2.morphologyEx(dark_object, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = largest_component_mask(dark_object)
    if np.mean(mask) < 0.01:
        return np.ones(gray.shape, dtype=bool)
    return mask


def bbox_from_mask(
    mask: np.ndarray,
    *,
    padding_ratio: float = 0.15,
    min_size_ratio: float = 0.20,
) -> tuple[int, int, int, int]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape[:2]
    if not np.any(mask):
        return 0, 0, width, height

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_width = x1 - x0
    box_height = y1 - y0
    min_width = int(round(width * min_size_ratio))
    min_height = int(round(height * min_size_ratio))
    box_width = max(box_width, min_width)
    box_height = max(box_height, min_height)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    pad = int(round(max(box_width, box_height) * padding_ratio))
    half_width = box_width // 2 + pad
    half_height = box_height // 2 + pad
    return (
        max(0, center_x - half_width),
        max(0, center_y - half_height),
        min(width, center_x + half_width),
        min(height, center_y + half_height),
    )


def crop_to_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    padding_ratio: float = 0.15,
    min_size_ratio: float = 0.20,
) -> np.ndarray:
    if mask is None or not np.any(mask):
        mask = foreground_mask_bgr(image_bgr)
        padding_ratio = max(padding_ratio, 0.08)
        min_size_ratio = max(min_size_ratio, 0.60)
    else:
        mask = largest_component_mask(mask)
    x0, y0, x1, y1 = bbox_from_mask(mask, padding_ratio=padding_ratio, min_size_ratio=min_size_ratio)
    return image_bgr[y0:y1, x0:x1].copy()


def align_crop_bgr(image_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Deskew elongated object crops; keep nearly square/texture crops unchanged."""
    if mask is None or not np.any(mask):
        mask = foreground_mask_bgr(image_bgr)
    mask = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return image_bgr
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    (_, _), (box_width, box_height), angle = rect
    long_side = max(box_width, box_height)
    short_side = max(min(box_width, box_height), 1.0)
    if long_side / short_side < 1.4:
        return image_bgr
    if box_width < box_height:
        angle += 90
    if abs(angle) < 3 or abs(angle) > 45:
        return image_bgr
    center = (image_bgr.shape[1] / 2.0, image_bgr.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (image_bgr.shape[1], image_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def resize_with_padding(
    image_bgr: np.ndarray,
    size: tuple[int, int] = (320, 320),
    *,
    fill_value: int = 245,
) -> np.ndarray:
    target_width, target_height = size
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image crop.")
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), fill_value, dtype=np.uint8)
    x0 = (target_width - resized_width) // 2
    y0 = (target_height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def enhance_contrast_bgr(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def prepare_classifier_view(
    image_bgr: np.ndarray,
    defect_mask: np.ndarray | None = None,
    *,
    image_size: int = 320,
    align_object: bool = True,
    contrast_enhance: bool = True,
) -> np.ndarray:
    """Create a cleaner classifier input using the defect mask when available."""
    if defect_mask is not None and defect_mask.shape[:2] != image_bgr.shape[:2]:
        defect_mask = cv2.resize(
            defect_mask.astype(np.uint8),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    crop = crop_to_mask(image_bgr, defect_mask, padding_ratio=0.20, min_size_ratio=0.24)
    if align_object:
        crop_mask = foreground_mask_bgr(crop)
        crop = align_crop_bgr(crop, crop_mask)
    crop = resize_with_padding(crop, (image_size, image_size))
    return enhance_contrast_bgr(crop) if contrast_enhance else crop


def multiscale_gray_views(
    image_bgr: np.ndarray,
    *,
    sizes: tuple[int, ...] = (256, 320, 384),
    crop_object: bool = False,
    align_object: bool = False,
) -> list[np.ndarray]:
    base = crop_to_mask(image_bgr) if crop_object else image_bgr
    if align_object:
        base = align_crop_bgr(base)
    views = []
    for size in sizes:
        resized = resize_with_padding(base, (size, size)) if crop_object else cv2.resize(base, (size, size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        views.append(denoised.astype(np.float32))
    return views
