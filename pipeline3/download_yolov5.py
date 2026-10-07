import subprocess
import urllib.request
from pathlib import Path

import torch


def ensure_yolov5_repo(repo_dir: str | Path = "third_party/yolov5") -> str:
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        print(f"Cloning YOLOv5 into {repo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5", str(repo_dir)],
            check=True,
        )
    return str(repo_dir)


def download_yolov5_weights(model_name: str = "yolov5s", output_dir: str | Path = "weights") -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{model_name}.pt"

    if not weights_path.exists():
        url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{model_name}.pt"
        print(f"Downloading {url} -> {weights_path}")
        urllib.request.urlretrieve(url, str(weights_path))

    return str(weights_path)


def load_yolov5(weights_path: str, model_name: str = "yolov5s", repo_dir: str | Path = "third_party/yolov5"):
    repo_dir = ensure_yolov5_repo(repo_dir)
    model = torch.hub.load(repo_dir, "custom", path=weights_path, source="local", force_reload=False)
    model.eval()
    return model


if __name__ == "__main__":
    repo_dir = ensure_yolov5_repo()
    weights_path = download_yolov5_weights(model_name="yolov5s", output_dir="weights")
    print(f"YOLOv5 repo: {repo_dir}")
    print(f"YOLOv5 weights: {weights_path}")
