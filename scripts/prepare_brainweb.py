from __future__ import annotations

import csv
import gzip
import hashlib
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


SOURCE_DIR = Path(
    r"C:\Users\mohdrei\Documents\Machine Learning\assignment\BrainWeb Dataset"
)

OUTPUT_ROOT = Path("data/brainweb")
SPLIT_MANIFEST = OUTPUT_ROOT / "split_manifest.csv"

NOISE_LEVELS = (0.03, 0.06, 0.10)
BASE_SEED = 42


def read_split_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 10:
        raise ValueError(
            f"Expected 10 subjects in {path}; found {len(rows)}."
        )

    ids = [row["subject_id"] for row in rows]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate subject IDs found in split manifest.")

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val", "test")
    }

    if counts != {"train": 6, "val": 2, "test": 2}:
        raise ValueError(f"Expected a 6/2/2 split; found {counts}.")

    return rows


def noise_seed(subject_id: str, sigma: float) -> int:
    """Return a stable, reproducible seed for one subject/noise level."""
    text = f"{BASE_SEED}:{subject_id}:{sigma:.2f}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little")


def load_minc_gz(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Decompress MINC to a temporary file, then read its scaled values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "volume.mnc"

        with gzip.open(path, "rb") as source:
            with temp_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

        image = nib.load(str(temp_path))
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

        # Load the data before the temporary MINC file is removed.
        volume = image.get_fdata(dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}: {path}")

    if not np.isfinite(volume).all():
        raise ValueError(f"Non-finite values found in {path}")

    return volume, spacing


def main() -> None:
    rows = read_split_manifest(SPLIT_MANIFEST)
    metadata_rows = []

    for row in rows:
        subject_id = row["subject_id"]
        split = row["split"]

        source_path = SOURCE_DIR / f"{subject_id}_t1w_p4.mnc.gz"

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        volume, spacing = load_minc_gz(source_path)

        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-4):
            raise ValueError(
                f"{subject_id}: expected 1 mm spacing; found {spacing}"
            )

        minimum = float(volume.min())
        scale = float(volume.max())

        if minimum < 0 or scale <= 0:
            raise ValueError(
                f"{subject_id}: per-volume maximum normalization expects "
                f"non-negative values and a positive maximum; "
                f"found min={minimum}, max={scale}."
            )

        # Treat the supplied source as the reference, as agreed.
        reference = (volume / scale).astype(np.float32)

        for sigma in NOISE_LEVELS:
            level_name = f"sigma_{int(round(sigma * 100)):02d}"
            dataset_root = OUTPUT_ROOT / level_name

            noisy_dir = dataset_root / split / "noisy"
            clean_dir = dataset_root / split / "clean"

            noisy_dir.mkdir(parents=True, exist_ok=True)
            clean_dir.mkdir(parents=True, exist_ok=True)

            seed = noise_seed(subject_id, sigma)
            rng = np.random.default_rng(seed)

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=reference.shape,
            ).astype(np.float32)

            # Preserve the additive Gaussian values on disk.
            # The current NGPS load_volume() clips them when loaded.
            noisy = reference + noise

            filename = f"{subject_id}.npy"

            np.save(noisy_dir / filename, noisy)
            np.save(clean_dir / filename, reference)

            metadata_rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "source_file": source_path.name,
                    "shape": str(tuple(volume.shape)),
                    "spacing_mm": str(spacing),
                    "source_min": minimum,
                    "normalization_max": scale,
                    "normalization": "divide_by_subject_max",
                    "sigma": sigma,
                    "noise_seed": seed,
                    "noisy_clipped_on_disk": False,
                }
            )

        print(
            f"Prepared {subject_id:10s} split={split:5s} "
            f"shape={volume.shape} spacing={spacing}"
        )

    metadata_path = OUTPUT_ROOT / "preparation_metadata.csv"

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nSaved preparation metadata to {metadata_path}")
    print("Prepared separate roots: sigma_03, sigma_06, sigma_10")


if __name__ == "__main__":
    main()