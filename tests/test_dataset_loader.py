import pytest

from ml.dataset_loader import load_bottle_dataframe


def test_dataset_loader_finds_mvtec_bottle_records():
    dataset = load_bottle_dataframe()
    if dataset.empty:
        pytest.skip("MVTec bottle dataset is not included in the GitHub upload")

    assert len(dataset) == 292
    assert set(dataset["label"]) == {"good", "broken_large", "broken_small", "contamination"}
    assert dataset[dataset["is_defective"] & dataset["mask_path"].isna()].empty
