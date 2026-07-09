import numpy as np

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
