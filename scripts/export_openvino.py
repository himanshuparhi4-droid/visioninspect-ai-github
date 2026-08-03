import json
import logging
import sys
from pathlib import Path

from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

# Add backend directory to path so we can import from ml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import SUPPORTED_CATEGORIES, _registry_overrides, category_model_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def export_models():
    registry = _registry_overrides()

    for category in SUPPORTED_CATEGORIES:
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
                input_size=(256, 256)
            )
            logger.info(f"Successfully exported {category} to {exported_path}")

            if category not in registry:
                registry[category] = {}
            registry[category]["openvino_path"] = str(exported_path)

        except Exception as e:
            logger.error(f"Failed to export {category}: {e}")

    # Save updated registry
    registry_path = PROJECT_ROOT / "models" / "category_model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    logger.info("Export process complete.")

if __name__ == "__main__":
    export_models()
