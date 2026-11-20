import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# This script is located in NGPS-master/scripts.
PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL_PROJECT = PROJECT.parent / "NGPS-master-original"


def main():
    parser = argparse.ArgumentParser(
        description="Compare clean, noisy, and denoised BrainWeb images."
    )

    parser.add_argument(
        "--sigma",
        choices=["03", "06", "10"],
        required=True,
        help="Noise level: 03, 06, or 10.",
    )

    parser.add_argument(
        "--subject",
        choices=["04", "06"],
        required=True,
        help="Test subject: 04 or 06.",
    )

    parser.add_argument(
        "--slice",
        type=int,
        default=90,
        help="Slice index to display. Default: 90.",
    )

    args = parser.parse_args()

    sigma = args.sigma
    subject = args.subject
    slice_index = args.slice

    # Radius 1 at sigma 0.03 uses a different output folder.
    radius1_folder = (
        "verified_radius1_sigma03"
        if sigma == "03"
        else f"radius1_sigma{sigma}"
    )

    image_paths = [
        (
            "Clean reference",
            PROJECT
            / "data"
            / "brainweb"
            / f"sigma_{sigma}"
            / "test"
            / "clean"
            / f"subject{subject}.npy",
        ),
        (
            "Noisy input",
            PROJECT
            / "data"
            / "brainweb"
            / f"sigma_{sigma}"
            / "test"
            / "noisy"
            / f"subject{subject}.npy",
        ),
        (
            "Original NGPS",
            ORIGINAL_PROJECT
            / "outputs"
            / f"original_sigma{sigma}"
            / f"subject{subject}_denoised.npy",
        ),
        (
            "NS-N2N",
            PROJECT
            / "outputs"
            / f"nsn2n_sigma{sigma}"
            / f"subject{subject}_denoised.npy",
        ),
        (
            "Same-location",
            PROJECT
            / "outputs"
            / f"samelocation_sigma{sigma}"
            / f"subject{subject}_denoised.npy",
        ),
        (
            "Adaptive radius 1",
            PROJECT
            / "outputs"
            / radius1_folder
            / f"subject{subject}_denoised.npy",
        ),
        (
            "Adaptive radius 2",
            PROJECT
            / "outputs"
            / f"radius2_sigma{sigma}"
            / f"subject{subject}_denoised.npy",
        ),
    ]

    missing_files = [
        path for _, path in image_paths if not path.exists()
    ]

    if missing_files:
        print("\nThe following files were not found:")
        for path in missing_files:
            print(f"  {path}")

        raise FileNotFoundError(
            "Check the paths and filenames shown above."
        )

    # Load each volume.
    volumes = [
        (name, np.load(path))
        for name, path in image_paths
    ]

    reference_shape = volumes[0][1].shape

    for name, volume in volumes:
        if volume.ndim != 3:
            raise ValueError(
                f"{name} must be a 3D volume; got {volume.shape}."
            )

        if volume.shape != reference_shape:
            raise ValueError(
                f"{name} has shape {volume.shape}; "
                f"expected {reference_shape}."
            )

    if not 0 <= slice_index < reference_shape[0]:
        raise ValueError(
            f"Slice index must be between 0 and {reference_shape[0] - 1}."
        )

    sigma_labels = {
        "03": "0.03",
        "06": "0.06",
        "10": "0.10",
    }

    # Six grid columns allow two images to be centered in rows 1 and 3.
    fig = plt.figure(figsize=(12, 12))

    grid = fig.add_gridspec(
        3,
        6,
        hspace=0.18,
        wspace=0.10,
    )

    positions = [
        grid[0, 1:3],  # Clean reference
        grid[0, 3:5],  # Noisy input

        grid[1, 0:2],  # Original NGPS
        grid[1, 2:4],  # NS-N2N
        grid[1, 4:6],  # Same-location

        grid[2, 1:3],  # Adaptive radius 1
        grid[2, 3:5],  # Adaptive radius 2
    ]

    for (name, volume), position in zip(volumes, positions):
        ax = fig.add_subplot(position)

        ax.imshow(
            volume[slice_index],
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_title(name, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        f"BrainWeb subject {subject} | "
        f"σ = {sigma_labels[sigma]} | "
        f"slice index = {slice_index}",
        fontsize=16,
        y=0.98,
    )

    output_dir = PROJECT / "outputs" / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"subject{subject}_sigma{sigma}_slice{slice_index}_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Comparison saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()