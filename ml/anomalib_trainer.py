from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.pre_processing import PreProcessor
from torchvision.transforms import v2

from ml.config import MVTEC_DATASET_ROOT


def get_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def build_mvtec_datamodule(
    root: str | Path = MVTEC_DATASET_ROOT,
    category: str = "bottle",
    image_size: tuple[int, int] = (256, 256),
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 0,
    apply_resize_augmentation: bool = True,
) -> MVTecAD:
    transform = v2.Compose([v2.Resize(image_size, antialias=True)]) if apply_resize_augmentation else None
    return MVTecAD(
        root=root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        augmentations=transform,
    )


def build_inspection_preprocessor(image_size: tuple[int, int] = (256, 256), roi_scale: float = 1.0) -> PreProcessor:
    """Create the same resize/center-ROI transform for training and live inference."""
    if roi_scale < 1:
        raise ValueError("roi_scale must be at least 1.0")
    if roi_scale == 1:
        transform = v2.Resize(image_size, antialias=True)
    else:
        expanded_size = tuple(int(round(value * roi_scale)) for value in image_size)
        transform = v2.Compose([v2.Resize(expanded_size, antialias=True), v2.CenterCrop(image_size)])
    return PreProcessor(transform=transform)


def build_anomalib_model(
    model_name: str = "padim",
    padim_n_features: int = 256,
    image_size: tuple[int, int] = (256, 256),
    roi_scale: float = 1.0,
    backbone: str | None = None,
    coreset_sampling_ratio: float = 0.05,
    num_neighbors: int = 5,
):
    model_name = model_name.lower()
    pre_processor = build_inspection_preprocessor(image_size=image_size, roi_scale=roi_scale)
    if model_name == "padim":
        return Padim(
            backbone=backbone or "resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            n_features=padim_n_features,
            pre_processor=pre_processor,
        )
    if model_name == "patchcore":
        return Patchcore(
            backbone=backbone or "resnet18",
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            pre_processor=pre_processor,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_engine(
    output_dir: str | Path,
    accelerator: str | None = None,
    devices: int = 1,
    max_epochs: int = 1,
    logger: bool = False,
    **trainer_kwargs,
) -> Engine:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    return Engine(
        accelerator=accelerator or get_accelerator(),
        devices=devices,
        max_epochs=max_epochs,
        default_root_dir=Path(output_dir),
        logger=logger,
        **trainer_kwargs,
    )
