from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


ASSIGNMENT = Path(__file__).resolve().parents[2]

ADAPTIVE_PROJECT = ASSIGNMENT / "NGPS-master"
ORIGINAL_PROJECT = ASSIGNMENT / "NGPS-master-original"

SIGMAS = ("03", "06", "10")
SUBJECTS = ("04", "06")


def checkpoint_path(method: str, sigma: str) -> Path:
    if method == "Original NGPS":
        return (
            ORIGINAL_PROJECT
            / "runs"
            / f"original_sigma{sigma}"
            / "model_best.pth"
        )

    if method == "NS-N2N":
        return (
            ADAPTIVE_PROJECT
            / "runs"
            / f"nsn2n_sigma{sigma}"
            / "model_best.pth"
        )

    if method == "Adaptive radius 1":
        return (
            ADAPTIVE_PROJECT
            / "runs"
            / f"Adaptive_brainweb_sigma_{sigma}_radius_1"
            / "model_best.pth"
        )

    if method == "Adaptive radius 2":
        return (
            ADAPTIVE_PROJECT
            / "runs"
            / f"Adaptive_brainweb_sigma_{sigma}_radius_2"
            / "model_best.pth"
        )

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    methods = (
        "Original NGPS",
        "NS-N2N",
        "Adaptive radius 1",
        "Adaptive radius 2",
    )

    output_root = ADAPTIVE_PROJECT / "outputs" / "inference_timing"
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "inference_times.csv"

    fieldnames = [
        "method",
        "sigma",
        "subject",
        "seconds",
        "status",
        "checkpoint",
    ]

    # Write each result immediately so completed timings remain
    # available if a later run is interrupted.
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for method in methods:
            for sigma in SIGMAS:
                for subject in SUBJECTS:
                    checkpoint = checkpoint_path(method, sigma)

                    input_path = (
                        ADAPTIVE_PROJECT
                        / "data"
                        / "brainweb"
                        / f"sigma_{sigma}"
                        / "test"
                        / "noisy"
                        / f"subject{subject}.npy"
                    )

                    # Run original NGPS with its own project code.
                    if method == "Original NGPS":
                        project = ORIGINAL_PROJECT
                    else:
                        project = ADAPTIVE_PROJECT

                    script = project / "scripts" / "infer_volume.py"

                    safe_method = (
                        method.lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                    )

                    output_path = (
                        output_root
                        / safe_method
                        / f"sigma_{sigma}"
                        / f"subject{subject}_denoised.npy"
                    )

                    row = {
                        "method": method,
                        "sigma": f"0.{sigma}",
                        "subject": f"subject{subject}",
                        "seconds": "",
                        "status": "",
                        "checkpoint": str(checkpoint),
                    }

                    missing = [
                        path
                        for path in (checkpoint, input_path, script)
                        if not path.is_file()
                    ]

                    if missing:
                        row["status"] = "missing file"
                        print("\nMissing:")
                        for path in missing:
                            print(f"  {path}")

                        writer.writerow(row)
                        file.flush()
                        continue

                    command = [
                        sys.executable,
                        str(script),
                        "--checkpoint",
                        str(checkpoint),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "cuda",
                    ]

                    print(
                        f"\nTiming {method}, sigma 0.{sigma}, "
                        f"subject {subject}"
                    )

                    start = time.perf_counter()

                    result = subprocess.run(
                        command,
                        cwd=project,
                        check=False,
                    )

                    elapsed = time.perf_counter() - start

                    row["seconds"] = f"{elapsed:.3f}"
                    row["status"] = (
                        "completed" if result.returncode == 0
                        else f"failed: exit {result.returncode}"
                    )

                    writer.writerow(row)
                    file.flush()

                    print(
                        f"Elapsed time: {elapsed:.3f} seconds "
                        f"({row['status']})"
                    )

    print(f"\nTiming results saved to:\n{csv_path}")


if __name__ == "__main__":
    main()