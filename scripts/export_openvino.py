import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import openvino as ov
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

# Add backend directory to path so we can import from ml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import SUPPORTED_CATEGORIES, _registry_overrides, category_model_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def compress_openvino_model(model_path: Path) -> dict:
    """Rewrite an OpenVINO graph with FP16 constants and report its final size."""
    core = ov.Core()
    model = core.read_model(str(model_path))
    bin_path = model_path.with_suffix(".bin")
    temporary_xml = model_path.with_name(f"{model_path.stem}.fp16.xml")
    temporary_bin = temporary_xml.with_suffix(".bin")
    ov.save_model(model, str(temporary_xml), compress_to_fp16=True)
    del model, core
    gc.collect()
    bin_path.unlink(missing_ok=True)
    temporary_bin.replace(bin_path)
    temporary_xml.replace(model_path)
    return {
        "precision": "FP16",
        "xml_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "bin_size_mb": round(bin_path.stat().st_size / (1024 * 1024), 2),
    }


def export_models(categories: list[str] | None = None, *, compress_fp16: bool = True):
    registry = _registry_overrides()

    selected_categories = categories or list(SUPPORTED_CATEGORIES)
    for category in selected_categories:
        spec = category_model_spec(category)
        if not spec.is_trained:
            logger.info(f"Skipping {category} - not trained")
            continue

        checkpoint_path = Path(spec.checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found for {category} at {checkpoint_path}")
            continue

        model_kind = spec.model_kind.lower()

        logger.info(f"Exporting {category} ({model_kind}) to OpenVINO...")

        # We need to initialize the correct model architecture to load the checkpoint
        if model_kind == "padim":
            model = Padim.load_from_checkpoint(str(checkpoint_path), weights_only=False)
        elif model_kind == "patchcore":
            model = Patchcore.load_from_checkpoint(str(checkpoint_path), weights_only=False)
        else:
            logger.error(f"Unsupported model kind {model_kind} for {category}")
            continue

        # Export
        export_root = PROJECT_ROOT / "models" / "exported" / category
        export_root.mkdir(parents=True, exist_ok=True)

        engine = Engine()
        try:
            exported_path = engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=str(export_root),
                input_size=(spec.input_size, spec.input_size),
            )
            logger.info(f"Successfully exported {category} to {exported_path}")

            exported_path = Path(exported_path).resolve()
            export_metadata = compress_openvino_model(exported_path) if compress_fp16 else {"precision": "FP32"}
            logger.info("Prepared %s deployment artifact: %s", category, export_metadata)

            if category not in registry:
                registry[category] = {}
            registry[category]["openvino_path"] = str(
                exported_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/")

            metadata_path = spec.metadata_path
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
            )
            metadata["openvino_export"] = {
                **export_metadata,
                "input_size": [spec.input_size, spec.input_size],
                "path": registry[category]["openvino_path"],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        except Exception as e:
            logger.error(f"Failed to export {category}: {e}")

    # Save updated registry
    registry_path = PROJECT_ROOT / "models" / "category_model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    logger.info("Export process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained anomaly models to OpenVINO.")
    parser.add_argument(
        "--categories",
        default=",".join(SUPPORTED_CATEGORIES),
        help="Comma-separated categories to export.",
    )
    parser.add_argument("--fp32", action="store_true", help="Keep FP32 constants instead of compact FP16.")
    args = parser.parse_args()
    categories = [
        item.strip().lower().replace("-", "_")
        for item in args.categories.split(",")
        if item.strip()
    ]
    unsupported = sorted(set(categories) - set(SUPPORTED_CATEGORIES))
    if unsupported:
        raise ValueError(f"Unsupported categories: {', '.join(unsupported)}")
    export_models(categories, compress_fp16=not args.fp32)
