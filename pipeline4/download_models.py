import subprocess
import sys
from pathlib import Path


def install_package(package_name: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)


def ensure_easyocr() -> None:
    try:
        import easyocr  # noqa: F401
    except ImportError:
        install_package("easyocr")


def ensure_torchvision() -> None:
    try:
        import torchvision  # noqa: F401
    except ImportError:
        install_package("torchvision")


def ensure_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        install_package("transformers")


def download_ocr_reader():
    ensure_easyocr()
    from easyocr import Reader

    reader = Reader(["en"], gpu=True)
    return reader


def download_fasterrcnn_model():
    ensure_torchvision()
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=True)
    model.eval()
    return model, weights


def download_dino_model():
    ensure_transformers()
    from transformers import AutoImageProcessor, AutoModel

    model_name = "facebook/dinov2_vits14"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


if __name__ == "__main__":
    print("Downloading OCR model...")
    reader = download_ocr_reader()
    print(f"OCR model ready: {type(reader).__name__}")

    print("Downloading Faster R-CNN model...")
    model, weights = download_fasterrcnn_model()
    print(f"Faster R-CNN model ready: {type(model).__name__}, weights={weights}")

    print("Downloading DINO model...")
    processor, model = download_dino_model()
    print(f"DINO processor/model ready: {type(processor).__name__}, {type(model).__name__}")
