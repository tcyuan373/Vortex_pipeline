"""Sample images for the YOLOv5 benchmark, taken from the VisA industrial defect dataset.

VisA (Visual Anomaly dataset, Amazon) has 10,821 images of 12 kinds of manufactured
objects, 1,200 of them with defects. It is released under CC BY 4.0:
https://github.com/amazon-science/spot-diff

The full archive is one large tar file on S3. Instead of downloading all of it, this
module streams the tar and keeps only the images it needs from the requested category,
then stops. The selected images are cached in `output_dir`, so later runs reuse them
without touching the network.

Each category has 100 defective images and 500-1,000 defect-free ones. The default
label, "mixed", takes all the defective images (up to `num_images`) and fills the rest
with defect-free ones, spread evenly through the list so every stretch of images the
benchmark cycles through has a similar share of defects. With the default 200 images
that is 100 defective and 100 defect-free, alternating.

Archive layout (inside the tar):
    <category>/Data/Images/Anomaly/*.JPG   images with defects
    <category>/Data/Images/Normal/*.JPG    defect-free images
"""

import argparse
import os
import re
import tarfile
import urllib.request
from pathlib import Path

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_HOME = "https://github.com/amazon-science/spot-diff"
VISA_LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"

VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
VISA_LABELS = ("mixed", "anomaly", "normal", "any")
# Per the official split file, every VisA category has exactly 100 defective images.
ANOMALIES_PER_CATEGORY = 100

# Printed circuit boards are the closest VisA category to appliance electronics.
DEFAULT_CATEGORY = "pcb1"
DEFAULT_LABEL = "mixed"

_MEMBER_RE = re.compile(
    r"(?:^|/)(?P<category>[^/]+)/Data/Images/(?P<label>Anomaly|Normal)/(?P<file>[^/]+\.(?:jpe?g|png))$",
    re.IGNORECASE,
)


def _cache_prefix(category: str, label: str) -> str:
    return f"visa_{category}_{label}_"


def _cached_images(output_dir: Path, category: str, label: str) -> list[Path]:
    return sorted(output_dir.glob(_cache_prefix(category, label) + "*"))


def _open_visa_stream(tar_path: str | Path | None):
    """Return a readable binary stream over the VisA tar, from a local copy or S3."""
    tar_path = tar_path or os.environ.get("VISA_TAR")
    if tar_path:
        print(f"Reading VisA archive from {tar_path}")
        return open(tar_path, "rb")
    print(f"Streaming VisA archive from {VISA_URL}")
    return urllib.request.urlopen(VISA_URL, timeout=60)


def _write_attribution(output_dir: Path) -> None:
    files = sorted(p.name for p in output_dir.glob("visa_*"))
    lines = [
        "# VisA sample images",
        "",
        "The `visa_*` images in this folder are unmodified copies from the VisA dataset",
        "(Zou et al., \"SPot-the-Difference Self-supervised Pre-training for Anomaly",
        "Detection and Segmentation\", ECCV 2022).",
        "",
        f"- Source: {VISA_HOME}",
        f"- Archive: {VISA_URL}",
        f"- License: {VISA_LICENSE}",
        "",
        "Files:",
        "",
        *[f"- {name}" for name in files],
        "",
    ]
    (output_dir / "ATTRIBUTION_VisA.md").write_text("\n".join(lines))


def _interleave(anomalies: list, normals: list) -> list:
    """Spread `anomalies` evenly through `normals`, keeping each list's order."""
    total = len(anomalies) + len(normals)
    if not anomalies or not normals:
        return anomalies + normals
    merged, a, n = [], iter(anomalies), iter(normals)
    for k in range(total):
        # Take an anomaly whenever the running share of anomalies falls behind its overall share.
        if (k + 1) * len(anomalies) // total > k * len(anomalies) // total:
            merged.append(next(a))
        else:
            merged.append(next(n))
    return merged


def create_sample_images(
    output_dir: str | Path = "sample_images",
    num_images: int = 200,
    category: str = DEFAULT_CATEGORY,
    label: str = DEFAULT_LABEL,
    tar_path: str | Path | None = None,
) -> list[str]:
    """Return paths to `num_images` VisA images, downloading them on first use.

    Args:
        output_dir: Folder for the selected images (also used as the cache).
        num_images: How many distinct images to keep.
        category: One of VISA_CATEGORIES, e.g. "pcb1".
        label: "mixed" (all defective images, rest defect-free, interleaved),
            "anomaly" (defective only, at most 100), "normal" (defect-free only) or
            "any" (whatever comes first in the archive).
        tar_path: Optional local copy of VisA_20220922.tar. The VISA_TAR
            environment variable works too. Without either, the tar is streamed from S3.
    """
    if category not in VISA_CATEGORIES:
        raise ValueError(f"Unknown VisA category {category!r}. Choose from: {', '.join(VISA_CATEGORIES)}")
    if label not in VISA_LABELS:
        raise ValueError(f"Unknown label {label!r}. Choose from: {', '.join(VISA_LABELS)}")
    if num_images < 1:
        raise ValueError("num_images must be at least 1.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # How many images of each kind to keep. None means "whatever comes first".
    if label == "mixed":
        anomaly_quota = min(num_images, ANOMALIES_PER_CATEGORY)
        quotas = {"anomaly": anomaly_quota, "normal": num_images - anomaly_quota}
    elif label == "any":
        quotas = None
    else:
        quotas = {label: num_images}

    cached = _cached_images(output_dir, category, label)
    if label == "mixed":
        # Rebuild the mix from the cache, so a cache made for a different count
        # (e.g. 500 images, 1 in 5 defective) still yields the right proportions.
        by_kind = {kind: [p for p in cached if f"_{kind}_" in p.name] for kind in ("anomaly", "normal")}
        if all(len(by_kind[kind]) >= quota for kind, quota in quotas.items()):
            mixed = _interleave(by_kind["anomaly"][:quotas["anomaly"]], by_kind["normal"][:quotas["normal"]])
            return [str(p) for p in mixed]
    elif len(cached) >= num_images:
        return [str(p) for p in cached[:num_images]]

    # Not enough cached images for this selection: start this selection over,
    # including leftovers from an interrupted download.
    for old in cached + list(output_dir.glob(".visa_download_*.part")):
        old.unlink()

    # Download into temporary files first; final names are assigned once the order is known.
    collected: dict[str, list[tuple[Path, str, str]]] = {"anomaly": [], "normal": []}
    arrival: list[tuple[str, int]] = []  # (kind, index into collected[kind]) in archive order
    with _open_visa_stream(tar_path) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = _MEMBER_RE.search(member.name)
            if match is None or match["category"] != category:
                continue
            kind = match["label"].lower()
            if quotas is not None and len(collected[kind]) >= quotas.get(kind, 0):
                continue

            stem, ext = os.path.splitext(match["file"])
            # Build our own file name; never use the path stored in the archive.
            tmp_path = output_dir / f".visa_download_{len(arrival):04d}.part"
            tmp_path.write_bytes(archive.extractfile(member).read())
            collected[kind].append((tmp_path, stem, ext.lower()))
            arrival.append((kind, len(collected[kind]) - 1))
            if len(arrival) % 50 == 0:
                print(f"  downloaded {len(arrival)} images "
                      f"({len(collected['anomaly'])} defective, {len(collected['normal'])} defect-free)")

            if quotas is None and len(arrival) >= num_images:
                break
            if quotas is not None and all(len(collected[k]) >= q for k, q in quotas.items()):
                break

    if label == "mixed":
        order = _interleave([("anomaly", i) for i in range(len(collected["anomaly"]))],
                            [("normal", i) for i in range(len(collected["normal"]))])
    else:
        order = arrival

    saved: list[Path] = []
    for kind, i in order:
        tmp_path, stem, ext = collected[kind][i]
        out_path = output_dir / f"{_cache_prefix(category, label)}{len(saved):03d}_{kind}_{stem}{ext}"
        os.replace(tmp_path, out_path)
        saved.append(out_path)
    print(f"Saved {len(saved)} VisA {category} images "
          f"({len(collected['anomaly'])} defective, {len(collected['normal'])} defect-free) to {output_dir}")

    if not saved:
        raise RuntimeError(f"No VisA images found for category={category!r}, label={label!r}.")
    if len(saved) < num_images:
        print(f"Warning: only {len(saved)} images matched; the benchmark will reuse them to fill batches.")

    _write_attribution(output_dir)
    return [str(p) for p in saved]


def repeat_image_paths(image_paths: list[str], batch_size: int) -> list[str]:
    if not image_paths:
        raise ValueError("No sample images were found.")
    return [image_paths[i % len(image_paths)] for i in range(batch_size)]


def cycle_batch(image_paths: list[str], batch_size: int, batch_index: int) -> tuple[list[str], int]:
    """Return the `batch_index`-th consecutive batch of `batch_size` images, wrapping around.

    Batch k covers images [k*batch_size, (k+1)*batch_size) modulo the number of images,
    so consecutive batches walk through the whole set. Also returns the first index.
    """
    if not image_paths:
        raise ValueError("No sample images were found.")
    n = len(image_paths)
    start = (batch_index * batch_size) % n
    return [image_paths[(start + i) % n] for i in range(batch_size)], start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a few VisA defect-inspection images for benchmarking.")
    parser.add_argument("--output-dir", default="sample_images")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--category", default=DEFAULT_CATEGORY, choices=VISA_CATEGORIES)
    parser.add_argument("--label", default=DEFAULT_LABEL, choices=VISA_LABELS)
    parser.add_argument("--tar-path", default=None, help="Local copy of VisA_20220922.tar (optional)")
    args = parser.parse_args()

    paths = create_sample_images(
        output_dir=args.output_dir,
        num_images=args.num_images,
        category=args.category,
        label=args.label,
        tar_path=args.tar_path,
    )
    print(f"Sample images: {len(paths)} in {args.output_dir}")
