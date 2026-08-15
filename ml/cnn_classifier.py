from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ml.config import MODELS_DIR
from ml.object_preprocessing import enhance_contrast_bgr, prepare_classifier_view, read_bgr, resize_with_padding

CNN_ARTIFACT_TYPE = "visioninspect_resnet18_finetuned_classifier"
CNN_ONNX_ARTIFACT_TYPE = "visioninspect_resnet18_finetuned_classifier_onnx"
CNN_CLASSIFIER_FILE = "cnn_defect_classifier.pt"
CNN_CANDIDATE_FILE = "cnn_defect_classifier.candidate.pt"
CNN_ONNX_FILE = "cnn_defect_classifier.onnx"
CNN_ONNX_METADATA_FILE = "cnn_defect_classifier.json"
DEFAULT_RESNET_WEIGHTS_PATH = MODELS_DIR / "inference" / "resnet18-f37072fd.pth"


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_mask(mask_path: str | Path | None) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 0


class DefectCropDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        label_to_index: dict[str, int],
        *,
        image_size: int,
        train: bool,
        crop_mode: str,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.transform = build_transforms(image_size, train=train)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]
        image_bgr = read_bgr(row["image_path"])
        mask = read_mask(row.get("mask_path"))
        view = prepare_cnn_view(image_bgr, mask, image_size=self.image_size, crop_mode=self.crop_mode)
        image_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        label = self.label_to_index[str(row["label"])]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def build_transforms(image_size: int, *, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.55),
                transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def prepare_cnn_view(
    image_bgr: np.ndarray,
    defect_mask: np.ndarray | None,
    *,
    image_size: int,
    crop_mode: str,
) -> np.ndarray:
    if crop_mode == "full":
        return enhance_contrast_bgr(resize_with_padding(image_bgr, (image_size, image_size)))
    if crop_mode == "object":
        return prepare_classifier_view(image_bgr, None, image_size=image_size)
    if crop_mode in {"defect", "object_crop_or_anomaly_mask_crop"}:
        return prepare_classifier_view(image_bgr, defect_mask, image_size=image_size)
    raise ValueError(f"Unsupported CNN crop_mode: {crop_mode}")


def build_resnet18_classifier(num_classes: int, *, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=None)
    if DEFAULT_RESNET_WEIGHTS_PATH.exists():
        state = torch.load(DEFAULT_RESNET_WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features, num_classes))
    return model


def class_weights(labels: list[str], label_to_index: dict[str, int]) -> torch.Tensor:
    counts = pd.Series(labels).value_counts()
    weights = torch.ones(len(label_to_index), dtype=torch.float32)
    total = float(len(labels))
    for label, index in label_to_index.items():
        weights[index] = total / max(len(label_to_index) * float(counts.get(label, 1)), 1.0)
    return weights


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    labels: list[str],
    device: torch.device,
) -> tuple[dict, list[int], list[int]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, batch_targets in loader:
            images = images.to(device)
            logits = model(images)
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            targets.extend(batch_targets.tolist())

    metrics = {
        "accuracy": round(float(accuracy_score(targets, predictions)), 4),
        "macro_f1": round(float(f1_score(targets, predictions, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(labels)))).tolist(),
    }
    return metrics, targets, predictions


def train_cnn_defect_classifier(
    dataset_df: pd.DataFrame,
    output_path: str | Path,
    *,
    category: str,
    image_size: int = 320,
    epochs: int = 18,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    random_state: int = 42,
    patience: int = 5,
    label_order: list[str] | None = None,
    crop_mode: str = "defect",
) -> dict:
    data = dataset_df[dataset_df["label"] != "good"].copy()
    labels = label_order or sorted(data["label"].unique().tolist())
    labels = [label for label in labels if label in set(data["label"])]
    data = data[data["label"].isin(labels)].copy()
    if data.empty:
        raise ValueError("No labelled defect images found for CNN classifier training.")
    counts = data["label"].value_counts()
    if (counts < 2).any():
        raise ValueError("At least two labelled images per defect subtype are required.")

    development_index, test_index = train_test_split(
        np.arange(len(data)),
        test_size=0.20,
        stratify=data["label"].to_numpy(),
        random_state=random_state,
    )
    relative_train_index, relative_validation_index = train_test_split(
        np.arange(len(development_index)),
        test_size=0.25,
        stratify=data.iloc[development_index]["label"].to_numpy(),
        random_state=random_state + 1,
    )
    train_index = development_index[relative_train_index]
    validation_index = development_index[relative_validation_index]
    train_df = data.iloc[train_index].reset_index(drop=True)
    validation_df = data.iloc[validation_index].reset_index(drop=True)
    development_df = data.iloc[development_index].reset_index(drop=True)
    test_df = data.iloc[test_index].reset_index(drop=True)
    label_to_index = {label: index for index, label in enumerate(labels)}

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    train_dataset = DefectCropDataset(
        train_df,
        label_to_index,
        image_size=image_size,
        train=True,
        crop_mode=crop_mode,
    )
    validation_dataset = DefectCropDataset(
        validation_df,
        label_to_index,
        image_size=image_size,
        train=False,
        crop_mode=crop_mode,
    )
    test_dataset = DefectCropDataset(
        test_df,
        label_to_index,
        image_size=image_size,
        train=False,
        crop_mode=crop_mode,
    )
    loader_generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(device_name())
    model = build_resnet18_classifier(len(labels)).to(device)
    weights = class_weights(train_df["label"].tolist(), label_to_index).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_metrics = {"accuracy": 0.0, "macro_f1": 0.0}
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        metrics, _, _ = evaluate_model(model, validation_loader, labels=labels, device=device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = round(float(np.mean(losses)), 4) if losses else 0.0
        history.append(metrics)
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    # Refit from the same pretrained initialization on train + validation for
    # the selected epoch count. The held-out test images remain untouched.
    selected_epochs = max(best_epoch, 1)
    torch.manual_seed(random_state + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state + 2)
    development_dataset = DefectCropDataset(
        development_df,
        label_to_index,
        image_size=image_size,
        train=True,
        crop_mode=crop_mode,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(random_state + 2),
    )
    model = build_resnet18_classifier(len(labels)).to(device)
    development_weights = class_weights(development_df["label"].tolist(), label_to_index).to(device)
    criterion = nn.CrossEntropyLoss(weight=development_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=selected_epochs)
    for _ in range(selected_epochs):
        model.train()
        for images, targets in development_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    best_state = deepcopy(model.state_dict())
    final_metrics, y_eval, y_pred = evaluate_model(model, test_loader, labels=labels, device=device)
    final_metrics.update(
        {
            "labels": labels,
            "train_size": int(len(train_df)),
            "validation_size": int(len(validation_df)),
            "production_train_size": int(len(development_df)),
            "eval_size": int(len(test_df)),
            "feature_extractor": "resnet18_finetuned",
            "feature_mode": "cnn_heatmap_object_crop",
            "defect_only": True,
            "best_epoch": best_epoch,
            "history": history,
            "evaluation": {
                "protocol": (
                    "stratified train/validation/test evaluation; best epoch selected on validation, "
                    "model refitted on train plus validation, and final metrics measured once on untouched test data"
                ),
                "selected_classifier": "resnet18_finetuned_weighted_cross_entropy",
                "loss": "weighted_cross_entropy_label_smoothing_0.05",
                "augmentation": "horizontal/vertical flips, small affine, and brightness/contrast jitter",
                "best_validation_accuracy": round(float(best_metrics.get("accuracy", 0.0)), 4),
                "best_validation_macro_f1": round(float(best_metrics.get("macro_f1", 0.0)), 4),
            },
            "dataset_context": {
                "source": "MVTec AD labelled test folders",
                "protocol": "defect-only subtype CNN classification using object/defect crops",
                "category": category,
                "crop_mode": crop_mode,
            },
        }
    )

    artifact = {
        "artifact_type": CNN_ARTIFACT_TYPE,
        "category": category,
        "labels": labels,
        "label_to_index": label_to_index,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "image_size": [image_size, image_size],
        "metrics": final_metrics,
        "preprocessing": {
            "view": crop_mode,
            "alignment": "elongated_object_deskew",
            "contrast": "LAB_CLAHE",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return {
        "metrics": final_metrics,
        "output_path": output_path,
        "y_eval": y_eval,
        "y_pred": y_pred,
    }


@lru_cache(maxsize=16)
def load_cnn_classifier(path_value: str) -> tuple[nn.Module, dict, torch.device]:
    path = Path(path_value)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != CNN_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported CNN classifier artifact: {path}")
    labels = artifact["labels"]
    device = torch.device(device_name())
    model = build_resnet18_classifier(len(labels), freeze_backbone=False)
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()
    return model, artifact, device


def export_cnn_classifier_onnx(
    artifact_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Export a trained CNN classifier and its runtime metadata to ONNX."""
    artifact_path = Path(artifact_path)
    model, artifact, _ = load_cnn_classifier(str(artifact_path))
    model = model.cpu().eval()
    image_size = int(artifact["image_size"][0])
    output_path = Path(output_path) if output_path else artifact_path.with_name(CNN_ONNX_FILE)
    metadata_path = output_path.with_name(CNN_ONNX_METADATA_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        torch.zeros(1, 3, image_size, image_size, dtype=torch.float32),
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "artifact_type": CNN_ONNX_ARTIFACT_TYPE,
        "category": str(artifact["category"]),
        "labels": [str(label) for label in artifact["labels"]],
        "image_size": image_size,
        "preprocessing": artifact.get("preprocessing", {}),
        "classifier_engine": "fine_tuned_resnet18_onnx",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "model_path": output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def predict_cnn_defect_type(
    image_path: str | Path,
    artifact_path: str | Path,
    *,
    defect_mask: np.ndarray | None = None,
) -> dict:
    model, artifact, device = load_cnn_classifier(str(artifact_path))
    image_size = int(artifact["image_size"][0])
    crop_mode = str(artifact.get("preprocessing", {}).get("view", "defect"))
    image_bgr = read_bgr(image_path)
    view = prepare_cnn_view(image_bgr, defect_mask, image_size=image_size, crop_mode=crop_mode)
    image_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    tensor = build_transforms(image_size, train=False)(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    labels = [str(label) for label in artifact["labels"]]
    best_index = int(np.argmax(probabilities))
    return {
        "defect_type": labels[best_index],
        "confidence": round(float(probabilities[best_index]), 4),
        "class_probabilities": {
            label: round(float(probability), 4) for label, probability in zip(labels, probabilities, strict=True)
        },
        "classifier_engine": "fine_tuned_resnet18",
    }
