from pathlib import Path
import urllib.request


DEFAULT_IMAGE_URLS = [
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
]


def download_image(url: str, output_dir: str | Path, filename: str | None = None) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.rsplit("/", 1)[-1].split("?", 1)[0]

    output_path = output_dir / filename
    if not output_path.exists():
        urllib.request.urlretrieve(url, str(output_path))

    return str(output_path)


def create_sample_images(output_dir: str | Path = "sample_images", num_images: int = 2, image_urls: list[str] | None = None) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_urls = image_urls or DEFAULT_IMAGE_URLS
    image_paths: list[str] = []

    for i in range(num_images):
        url = image_urls[i % len(image_urls)]
        ext = ".jpg" if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg") else ".png"
        image_path = download_image(url, output_dir, filename=f"sample_{i}{ext}")
        image_paths.append(image_path)

    return image_paths


def repeat_image_paths(image_paths: list[str], batch_size: int) -> list[str]:
    if not image_paths:
        raise ValueError("No sample images were found.")
    return [image_paths[i % len(image_paths)] for i in range(batch_size)]


if __name__ == "__main__":
    paths = create_sample_images()
    print(f"Downloaded sample images: {paths}")
