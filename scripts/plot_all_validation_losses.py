from pathlib import Path
import csv

import matplotlib.pyplot as plt


# The script is located in NGPS-master/scripts.
PROJECT = Path(__file__).resolve().parents[1]
ASSIGNMENT = PROJECT.parent
ORIGINAL_PROJECT = ASSIGNMENT / "NGPS-master-original"


# Each sigma will occupy one row.
SIGMAS = [
    ("03", "0.03"),
    ("06", "0.06"),
    ("10", "0.10"),
]


# Each method will occupy one column.
METHODS = [
    (
        "Original NGPS",
        lambda sigma: (
            ORIGINAL_PROJECT
            / "runs"
            / f"original_sigma{sigma}"
            / "validation_original_ngps_loss.csv"
        ),
    ),
    (
        "NS-N2N",
        lambda sigma: (
            PROJECT
            / "runs"
            / f"nsn2n_sigma{sigma}"
            / "validation_nsn2n_loss.csv"
        ),
    ),
    (
        "Adaptive radius 1",
        lambda sigma: (
            PROJECT
            / "runs"
            / f"Adaptive_brainweb_sigma_{sigma}_radius_1"
            / "validation_adaptive_loss.csv"
        ),
    ),
    (
        "Adaptive radius 2",
        lambda sigma: (
            PROJECT
            / "runs"
            / f"Adaptive_brainweb_sigma_{sigma}_radius_2"
            / "validation_adaptive_loss.csv"
        ),
    ),
    (
        "Same-location",
        lambda sigma: (
            PROJECT
            / "runs"
            / f"samelocation_sigma{sigma}"
            / "validation_samelocation_loss.csv"
        ),
    ),
]


def read_mean_losses(csv_path):
    """Read mean validation loss for each saved epoch."""
    points = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["subject"].strip().upper() == "MEAN":
                epoch = int(row["epoch"])
                loss = float(row["validation_loss"])
                points.append((epoch, loss))

    if not points:
        raise ValueError(f"No MEAN rows found in: {csv_path}")

    return sorted(points)


def main():
    # Three rows for noise levels and five columns for methods.
    fig, axes = plt.subplots(
        nrows=len(SIGMAS),
        ncols=len(METHODS),
        figsize=(24, 13),
        squeeze=False,
    )

    fig.suptitle(
        "Validation Loss by Noise Level and Method",
        fontsize=22,
        y=0.98,
    )

    missing_files = []

    # Rows = sigmas.
    for row_index, (sigma_code, sigma_label) in enumerate(SIGMAS):

        # Columns = methods.
        for column_index, (method_name, path_for_sigma) in enumerate(METHODS):

            ax = axes[row_index, column_index]
            csv_path = path_for_sigma(sigma_code)

            ax.set_title(
                f"{method_name}, σ = {sigma_label}",
                fontsize=11,
                pad=10,
            )

            ax.set_ylabel(
                "Mean validation loss",
                fontsize=9,
            )

            # Keep epoch numbers but do not add the word "Epoch".
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=8)

            if not csv_path.exists():
                missing_files.append(csv_path)

                ax.text(
                    0.5,
                    0.5,
                    "CSV file not found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="red",
                )
                continue

            try:
                points = read_mean_losses(csv_path)

            except (KeyError, ValueError) as error:
                print(f"Could not read {csv_path}: {error}")

                ax.text(
                    0.5,
                    0.5,
                    "Could not read CSV",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="red",
                )
                continue

            epochs = [point[0] for point in points]
            losses = [point[1] for point in points]

            # Lowest loss among the available CSV entries.
            best_epoch, best_loss = min(
                points,
                key=lambda point: (point[1], point[0]),
            )

            ax.plot(
                epochs,
                losses,
                marker="o",
                markersize=4,
                color="steelblue",
                linewidth=1.7,
                label="Mean validation loss",
            )

            ax.scatter(
                best_epoch,
                best_loss,
                color="red",
                s=55,
                zorder=3,
                label=f"Selected epoch: {best_epoch}",
            )

            ax.set_xticks(epochs)

            ax.legend(
                fontsize=7,
                loc="best",
            )

    # Space between rows and columns.
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.06,
        top=0.92,
        hspace=0.55,
        wspace=0.40,
    )

    output = (
        PROJECT
        / "outputs"
        / "validation_losses_sigma_rows.png"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nGraph saved to: {output}")

    if missing_files:
        print("\nThe following CSV files were not found:")

        for path in missing_files:
            print(f"  {path}")

    plt.show()


if __name__ == "__main__":
    main()