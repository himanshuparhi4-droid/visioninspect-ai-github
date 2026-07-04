from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Fastflow, Padim, Patchcore
from torchvision.transforms import v2

from ml.config import MODELS_DIR, RAW_DATA_DIR


def get_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def build_mvtec_datamodule(
    root: str | Path = RAW_DATA_DIR.parent,
    category: str = "bottle",
    image_size: tuple[int, int] = (256, 256),
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 0,
) -> MVTecAD:
    transform = v2.Compose([v2.Resize(image_size, antialias=True)])
    return MVTecAD(
        root=root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        augmentations=transform,
    )


def build_anomalib_model(model_name: str = "padim"):
    model_name = model_name.lower()
    if model_name == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True, n_features=100)
    if model_name == "patchcore":
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=0.05,
            num_neighbors=5,
        )
    if model_name == "fastflow":
        return Fastflow(backbone="resnet18", pre_trained=True, flow_steps=8)
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_engine(
    output_dir: str | Path,
    accelerator: str | None = None,
    devices: int = 1,
    max_epochs: int = 1,
    logger: bool = False,
    **trainer_kwargs,
) -> Engine:
    return Engine(
        accelerator=accelerator or get_accelerator(),
        devices=devices,
        max_epochs=max_epochs,
        default_root_dir=Path(output_dir),
        logger=logger,
        **trainer_kwargs,
    )


def model_checkpoint_destination(model_name: str = "padim", version: str = "v1") -> Path:
    return MODELS_DIR / "checkpoints" / f"{model_name}_mvtec_bottle_{version}.ckpt"
