"""Sample images for the pipeline 4 benchmark: dairy-cow udder photos from Roboflow.

Source: "cow_udder" on Roboflow Universe (509 images, CC BY 4.0):
https://universe.roboflow.com/butter-vfw4d/cow_udder

Every image in the dataset is an udder photo (its labels mark teats), so no class
filtering is needed. The module exports the dataset through the Roboflow REST API,
picks `num_images` distinct photos (skipping Roboflow's augmented copies of the same
photo), resizes each to a uniform size, and writes an attribution file.

Uniform sizes keep Faster R-CNN from padding a whole batch up to its largest image,
which multiplied memory use when photos had mixed shapes. Images are scaled to fit
and padded with gray bars, so nothing is cropped or stretched.

Roboflow requires a free account: set ROBOFLOW_API_KEY (roboflow.com -> Settings ->
API Keys). Alternatively download the dataset in a browser (Download Dataset -> any
format -> zip) and set ROBOFLOW_ZIP=/path/to/that.zip; no key is needed then.

Downloaded originals are kept in `output_dir/.cowudder_originals/` and the resized
copies in `output_dir/cowudder_*.jpg`, so later runs, or a change of --size, reuse
them without touching the network.
"""

import argparse
import http.cookiejar
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

try:  # Roboflow's own SDK downloads with requests; use it when it is installed.
    import requests
except ImportError:
    requests = None

API_URL = "https://api.roboflow.com"
WORKSPACE = "butter-vfw4d"
PROJECT = "cow_udder"
DATASET_NAME = "cow_udder"
DATASET_PAGE = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}"
DATASET_LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"

# Export formats to try, in order. Each format is a separate zip on Roboflow's side, so
# if one format's file is missing from Roboflow's storage, another may still work.
EXPORT_FORMATS = ("coco", "yolov8", "yolov5pytorch")
# Without --version, the newest finished versions are tried, at most this many.
MAX_VERSIONS_TO_TRY = 3

CACHE_PREFIX = "cowudder_"
ORIGINALS_DIR = ".cowudder_originals"
ATTRIBUTION_FILE = "ATTRIBUTION_cow_udder.md"
SOURCES_FILE = "cow_udder_sources.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Size every benchmark image is resized to (width, height), and the padding color.
TARGET_SIZE = (1024, 768)
PAD_COLOR = (114, 114, 114)
# Prefer splits without augmentation; Roboflow only augments the train split.
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}


# ---------------------------------------------------------------- Roboflow API


def _get_json(url: str) -> tuple[int, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "vortex-pipeline-benchmark"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Don't echo the URL: it contains the API key.
        raise RuntimeError(f"Roboflow API returned HTTP {exc.code}: {body}") from None


def _api_url(path: str, api_key: str, **params) -> str:
    query = urllib.parse.urlencode({"api_key": api_key, **params})
    return f"{API_URL}/{path}?{query}"


def _version_ready(entry: dict) -> bool:
    """A version can be exported once it has images and is no longer being generated."""
    return int(entry.get("images") or 0) > 0 and not entry.get("generating")


def _ready_versions(api_key: str) -> list[int]:
    """Versions that have images and finished generating, newest first."""
    _, info = _get_json(_api_url(f"{WORKSPACE}/{PROJECT}", api_key))
    versions = info.get("versions", [])
    ready = sorted({int(v["id"].rsplit("/", 1)[-1]) for v in versions if _version_ready(v)}, reverse=True)
    if not ready:
        listed = [v.get("id") for v in versions]
        raise RuntimeError(f"No finished dataset versions with images for {WORKSPACE}/{PROJECT}. Listed: {listed}")
    return ready


def _wait_for_version(api_key: str, version: int, timeout_s: int = 600) -> None:
    """Wait while Roboflow is still generating `version`, as the official SDK does."""
    url = _api_url(f"{WORKSPACE}/{PROJECT}/{version}", api_key, nocache="true")
    deadline = time.time() + timeout_s
    while True:
        _, payload = _get_json(url)
        entry = payload.get("version", {})
        if _version_ready(entry):
            return
        if time.time() > deadline:
            raise RuntimeError(f"Version {version} is still generating or has no images.")
        print(f"  Roboflow is still generating version {version}: {float(entry.get('progress') or 0) * 100:.0f}%")
        time.sleep(5)


def _export_link(api_key: str, version: int, fmt: str, timeout_s: int = 600) -> str:
    """Ask Roboflow for an export of `version` in `fmt`, waiting while it is generated."""
    url = _api_url(f"{WORKSPACE}/{PROJECT}/{version}/{fmt}", api_key, nocache="true")
    deadline = time.time() + timeout_s
    while True:
        status, payload = _get_json(url)
        if status == 200 and "export" in payload:
            return payload["export"]["link"]
        if time.time() > deadline:
            raise RuntimeError(f"Roboflow export did not finish within {timeout_s}s: {payload}")
        progress = float(payload.get("progress") or 0.0)
        print(f"  Roboflow is preparing the export: {progress * 100:.0f}%")
        time.sleep(3)


class DownloadError(Exception):
    def __init__(self, code: int, chain: list[str]):
        self.code = code
        self.chain = chain
        super().__init__(f"HTTP {code} after {' -> '.join(chain)}")


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest`, keeping cookies across redirects like a browser does."""
    headers = {"User-Agent": "vortex-pipeline-benchmark"}
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        if requests is not None:
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                chain = [_host(r.url) for r in response.history] + [_host(response.url)]
                if response.status_code != 200:
                    raise DownloadError(response.status_code, chain)
                with open(tmp, "wb") as out:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        out.write(chunk)
        else:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            request = urllib.request.Request(url, headers=headers)
            try:
                with opener.open(request, timeout=120) as response, open(tmp, "wb") as out:
                    shutil.copyfileobj(response, out, length=1 << 20)
            except urllib.error.HTTPError as exc:
                raise DownloadError(exc.code, [_host(url), _host(exc.url or url)]) from None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)


class ExportMissing(RuntimeError):
    """Roboflow handed out a link to an export zip that is not in its storage bucket."""


def _download_export(api_key: str, version: int, fmt: str, dest: Path, attempts: int = 2) -> None:
    """Download one export zip, asking for a fresh link once if the first one 404s.

    A 404 from storage.googleapis.com means the zip the link points to does not exist
    in Roboflow's bucket: a problem on Roboflow's side, not a permissions error.
    """
    for attempt in range(1, attempts + 1):
        link = _export_link(api_key, version, fmt)
        try:
            _download(link, dest)
            return
        except DownloadError as exc:
            if exc.code == 404 and attempt == attempts:
                raise ExportMissing(f"version {version} {fmt} export file is missing from Roboflow's storage ({exc})") from None
            if exc.code != 404:
                raise RuntimeError(f"Downloading the version {version} {fmt} export failed: {exc}") from None
            print(f"  Export download returned {exc}; retrying with a fresh link...")
            time.sleep(10)


def _fetch_export(api_key: str, version: int | None, cache_dir: Path) -> tuple[Path, str]:
    """Download an export zip, falling back across formats (and versions, if not pinned)."""
    for existing in sorted(cache_dir.glob(f"{PROJECT}-v*.zip")):
        if version is None or existing.name.startswith(f"{PROJECT}-v{version}-"):
            print(f"Using cached export {existing}")
            return existing, existing.stem.removeprefix(f"{PROJECT}-")

    versions = [version] if version is not None else _ready_versions(api_key)[:MAX_VERSIONS_TO_TRY]
    failures: list[str] = []
    for v in versions:
        _wait_for_version(api_key, v)
        for fmt in EXPORT_FORMATS:
            zip_path = cache_dir / f"{PROJECT}-v{v}-{fmt}.zip"
            print(f"Exporting {WORKSPACE}/{PROJECT} version {v} ({fmt}) to {zip_path}...")
            try:
                _download_export(api_key, v, fmt, zip_path)
                return zip_path, f"v{v}-{fmt}"
            except ExportMissing as exc:
                failures.append(str(exc))
                print(f"  The {exc}; trying the next option.")
    raise RuntimeError(
        "Every Roboflow export tried was missing: " + "; ".join(failures) + ". "
        f"Download the dataset in a browser from {DATASET_PAGE} (Download Dataset -> zip) "
        "and rerun with ROBOFLOW_ZIP=/path/to/that.zip."
    )


# ------------------------------------------------------------ image selection


def _dedupe_key(member: str) -> str:
    """Roboflow names files '<original>_jpg.rf.<hash>.jpg'; augmented copies share <original>."""
    name = PurePosixPath(member).name
    return name.split(".rf.", 1)[0] if ".rf." in name else PurePosixPath(member).stem


def _pick_images(archive: zipfile.ZipFile, limit: int) -> list[str]:
    """Distinct photos from the export, preferring the non-augmented test/valid splits."""
    members = [
        m for m in archive.namelist()
        if PurePosixPath(m).suffix.lower() in IMAGE_EXTENSIONS and not m.startswith("__MACOSX/")
    ]

    def order(m: str) -> tuple[int, str]:
        parts = PurePosixPath(m).parts
        split = parts[0].lower() if len(parts) > 1 else ""
        return SPLIT_PRIORITY.get(split, 3), m

    picked: list[str] = []
    seen: set[str] = set()
    for m in sorted(members, key=order):
        key = _dedupe_key(m)
        if key not in seen:
            seen.add(key)
            picked.append(m)
        if len(picked) >= limit:
            break
    return picked


def _resize(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Scale `src` to fit `size` without cropping, pad the rest with gray, save as JPEG."""
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")  # honor camera rotation tags
        ImageOps.pad(img, size, method=Image.LANCZOS, color=PAD_COLOR).save(dst, "JPEG", quality=95)


def _write_attribution(output_dir: Path, records: list[dict], source_label: str, size: tuple[int, int]) -> None:
    lines = [
        "# Cow udder sample images",
        "",
        f"The `{CACHE_PREFIX}*` images in this folder are adapted from the \"{DATASET_NAME}\" dataset",
        f"by the {WORKSPACE} workspace on Roboflow Universe. Each photo was resized to fit",
        f"{size[0]}x{size[1]} pixels and the remainder padded with gray bars.",
        "",
        f"- Source: {DATASET_PAGE}",
        f"- Export: {source_label}",
        f"- License: {DATASET_LICENSE}",
        "",
        "| File | File in the Roboflow export |",
        "|---|---|",
        *[f"| {r['file']} | {r['source']} |" for r in records],
        "",
    ]
    (output_dir / ATTRIBUTION_FILE).write_text("\n".join(lines))
    (output_dir / SOURCES_FILE).write_text(
        json.dumps({"export": source_label, "size": list(size), "images": records}, indent=2)
    )


def _resize_from_originals(output_dir: Path, sources: dict, size: tuple[int, int]) -> list[Path]:
    """Re-create the resized copies of the current selection without downloading."""
    originals_dir = output_dir / ORIGINALS_DIR
    resized: list[Path] = []
    for record in sources["images"]:
        out_path = output_dir / record["file"]
        _resize(originals_dir / record["original_file"], out_path, size)
        resized.append(out_path)
    _write_attribution(output_dir, sources["images"], sources["export"], size)
    return resized


# ---------------------------------------------------------------------- main


def create_sample_images(
    output_dir: str | Path = "sample_images",
    num_images: int = 64,
    api_key: str | None = None,
    version: int | None = None,
    zip_path: str | Path | None = None,
    size: tuple[int, int] = TARGET_SIZE,
    keep_zip: bool = False,
) -> list[str]:
    """Return paths to up to `num_images` udder photos, all resized to `size` (width, height).

    Args:
        output_dir: Folder for the selected images (also used as the cache).
        num_images: How many distinct photos to keep.
        api_key: Roboflow API key. Defaults to the ROBOFLOW_API_KEY environment variable.
        version: Dataset version to export. Defaults to the newest version that downloads.
        zip_path: An export zip downloaded in a browser. ROBOFLOW_ZIP works too. Skips the API.
        size: Output size of every image, as (width, height).
        keep_zip: Keep the downloaded export zip in `output_dir/.roboflow_cache`.
    """
    if num_images < 1:
        raise ValueError("num_images must be at least 1.")
    size = (int(size[0]), int(size[1]))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = output_dir / ORIGINALS_DIR
    sources_file = output_dir / SOURCES_FILE

    # Reuse the current selection if it is big enough; re-resize it if the size changed.
    if sources_file.exists():
        sources = json.loads(sources_file.read_text())
        records = sources.get("images", [])
        have_originals = all((originals_dir / r["original_file"]).exists() for r in records)
        if len(records) >= num_images and have_originals:
            cached = [output_dir / r["file"] for r in records]
            if tuple(sources.get("size", ())) != size or not all(p.exists() for p in cached):
                print(f"Resizing cached images to {size[0]}x{size[1]}...")
                cached = _resize_from_originals(output_dir, sources, size)
            return [str(p) for p in cached[:num_images]]

    zip_path = zip_path or os.environ.get("ROBOFLOW_ZIP")
    if zip_path:
        zip_path = Path(zip_path)
        if not zip_path.is_file():
            raise FileNotFoundError(f"Roboflow export zip not found: {zip_path}")
        print(f"Using local Roboflow export {zip_path}")
        source_label = f"local file {zip_path.name}"
        keep_zip = True  # never delete a file the user provided
    else:
        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ROBOFLOW_API_KEY to download the cow udder images (a free Roboflow account "
                "gives you a key under Settings -> API Keys), or download the dataset zip from "
                f"{DATASET_PAGE} and set ROBOFLOW_ZIP=/path/to/that.zip."
            )
        cache_dir = output_dir / ".roboflow_cache"
        cache_dir.mkdir(exist_ok=True)
        zip_path, source_label = _fetch_export(api_key, version, cache_dir)

    # Replace any previous selection so the cache matches this run.
    originals_dir.mkdir(exist_ok=True)
    for old in list(output_dir.glob(CACHE_PREFIX + "*")) + list(originals_dir.glob(CACHE_PREFIX + "*")):
        old.unlink()

    records = []
    saved: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        picked = _pick_images(archive, num_images)
        if not picked:
            raise RuntimeError(f"No images found in {zip_path}.")
        for member in picked:
            ext = PurePosixPath(member).suffix.lower()
            stem = f"{CACHE_PREFIX}{len(saved):03d}"
            # Build our own file names; never use the path stored in the archive.
            original = originals_dir / f"{stem}{ext}"
            out_path = output_dir / f"{stem}.jpg"
            original.write_bytes(archive.read(member))
            try:
                _resize(original, out_path, size)
            except OSError as exc:
                print(f"  skipped {member}: {exc}")
                original.unlink(missing_ok=True)
                continue
            saved.append(out_path)
            records.append({"file": out_path.name, "original_file": original.name, "source": member})

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    print(f"Saved {len(saved)} images to {output_dir} ({size[0]}x{size[1]}).")
    if len(saved) < num_images:
        print(f"Warning: only {len(saved)} distinct photos found; the benchmark will reuse them to fill batches.")

    _write_attribution(output_dir, records, source_label, size)
    return [str(p) for p in saved]


def repeat_image_paths(image_paths: list[str], batch_size: int) -> list[str]:
    if not image_paths:
        raise ValueError("No sample images were found.")
    return [image_paths[i % len(image_paths)] for i in range(batch_size)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cow udder photos from Roboflow for benchmarking.")
    parser.add_argument("--output-dir", default="sample_images")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--version", type=int, default=None, help="Dataset version (default: newest that downloads)")
    parser.add_argument("--zip-path", default=None, help="Export zip downloaded in a browser (skips the API)")
    parser.add_argument("--size", default=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}", help="Output size WIDTHxHEIGHT")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded export zip")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    paths = create_sample_images(
        output_dir=args.output_dir,
        num_images=args.num_images,
        version=args.version,
        zip_path=args.zip_path,
        size=(width, height),
        keep_zip=args.keep_zip,
    )
    print(f"Sample images: {len(paths)} in {args.output_dir}")
