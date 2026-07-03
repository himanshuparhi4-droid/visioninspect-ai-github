import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.kfold_validation import prepare_classifier_dataframe


def test_prepare_classifier_dataframe_loads_mvtec_labels():
    dataset_root = Path("data/raw/mvtec_anomaly_detection/bottle")
    if not dataset_root.exists():
        pytest.skip("MVTec bottle dataset is not included in the GitHub upload")

    data = prepare_classifier_dataframe(dataset_root, max_images_per_class=2)

    assert not data.empty
    assert set(data["label"]).issubset({"good", "broken_large", "broken_small", "contamination"})
    assert data["image_path"].notna().all()


def test_prepare_classifier_dataframe_rejects_missing_dataset(tmp_path):
    with pytest.raises(ValueError):
        prepare_classifier_dataframe(tmp_path)
