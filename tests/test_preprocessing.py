import numpy as np

from ml.object_preprocessing import crop_to_mask, prepare_classifier_view
from ml.preprocessing import denoise_image, detect_edges, enhance_contrast, resize_image, to_grayscale


def test_preprocessing_pipeline_shapes_and_types():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[20:60, 35:90] = [40, 120, 220]

    resized = resize_image(image, size=(256, 256))
    gray = to_grayscale(resized)
    denoised = denoise_image(gray)
    enhanced = enhance_contrast(denoised)
    edges = detect_edges(enhanced)

    assert resized.shape == (256, 256, 3)
    assert gray.shape == (256, 256)
    assert denoised.shape == (256, 256)
    assert enhanced.shape == (256, 256)
    assert edges.shape == (256, 256)
    assert resized.dtype == np.uint8
    assert gray.dtype == np.uint8
    assert edges.dtype == np.uint8


def test_to_grayscale_keeps_existing_grayscale_image():
    gray = np.zeros((32, 32), dtype=np.uint8)

    result = to_grayscale(gray)

    assert result.shape == gray.shape
    assert np.array_equal(result, gray)


def test_object_preprocessing_uses_defect_mask_crop():
    image = np.full((120, 160, 3), 245, dtype=np.uint8)
    image[25:100, 45:120] = [70, 120, 180]
    mask = np.zeros((120, 160), dtype=bool)
    mask[55:70, 80:95] = True

    cropped = crop_to_mask(image, mask, padding_ratio=0.2)
    prepared = prepare_classifier_view(image, mask, image_size=96, align_object=False)

    assert cropped.shape[0] < image.shape[0]
    assert cropped.shape[1] < image.shape[1]
    assert prepared.shape == (96, 96, 3)
    assert prepared.dtype == np.uint8
