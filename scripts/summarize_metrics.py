from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# Both project folders are inside the assignment folder.
ASSIGNMENT = Path(__file__).resolve().parents[2]

# Folder names and their corresponding method and noise level.
FOLDERS = [
    ("NGPS-master-original", "original_sigma03", "Original NGPS", 0.03),
    ("NGPS-master-original", "original_sigma06", "Original NGPS", 0.06),
    ("NGPS-master-original", "original_sigma10", "Original NGPS", 0.10),

    ("NGPS-master", "nsn2n_sigma03", "NS-N2N", 0.03),
    ("NGPS-master", "nsn2n_sigma06", "NS-N2N", 0.06),
    ("NGPS-master", "nsn2n_sigma10", "NS-N2N", 0.10),

    ("NGPS-master", "radius1_sigma03", "Adaptive radius 1", 0.03),
    ("NGPS-master", "radius1_sigma06", "Adaptive radius 1", 0.06),
    ("NGPS-master", "radius1_sigma10", "Adaptive radius 1", 0.10),

    ("NGPS-master", "radius2_sigma03", "Adaptive radius 2", 0.03),
    ("NGPS-master", "radius2_sigma06", "Adaptive radius 2", 0.06),
    ("NGPS-master", "radius2_sigma10", "Adaptive radius 2", 0.10),

    ("NGPS-master", "samelocation_sigma03", "Same-location", 0.03),
    ("NGPS-master", "samelocation_sigma06", "Same-location", 0.06),
    ("NGPS-master", "samelocation_sigma10", "Same-location", 0.10),]

SUBJECTS = ("subject04", "subject06")
METRICS = ("PSNR", "SSIM", "FSIM", "HFEN", "GMSD")


def summarize_file(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    results = []
    missing = []

    for project, folder, method, sigma in FOLDERS:
        for subject in SUBJECTS:
            path = (
                ASSIGNMENT
                / project
                / "outputs"
                / folder
                / f"{subject}_metrics.csv"
            )

            if not path.is_file():
                missing.append(path)
                continue

            averages = summarize_file(path)

            results.append(
                {
                    "method": method,
                    "sigma": sigma,
                    "subject": subject,
                    **averages,
                }
            )

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  {path}")

    if not results:
        raise RuntimeError("No metric CSV files were found.")

    output = ASSIGNMENT / "NGPS-master" / "outputs" / "metrics_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "sigma", "subject", *METRICS]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} subject-level rows to:")
    print(output)
    print("Expected: 24 rows for 4 methods × 3 sigmas × 2 subjects.")


if __name__ == "__main__":
    main()