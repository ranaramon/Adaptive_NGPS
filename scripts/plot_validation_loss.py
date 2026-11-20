import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Validation Loss")
    args = parser.parse_args()

    epochs = []
    losses = []

    # Read the average validation loss for each epoch.
    with open(args.csv, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["subject"] == "MEAN":
                epochs.append(int(row["epoch"]))
                losses.append(float(row["validation_loss"]))

    if not epochs:
        raise ValueError("No MEAN rows found in the CSV file.")

    # Put the epochs in numerical order.
    points = sorted(zip(epochs, losses))
    epochs, losses = zip(*points)

    # Find the epoch with the lowest validation loss.
    best_index = min(range(len(losses)), key=lambda i: losses[i])
    best_epoch = epochs[best_index]
    best_loss = losses[best_index]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        losses,
        marker="o",
        linewidth=2,
        label="Mean validation loss",
    )

    plt.scatter(
        best_epoch,
        best_loss,
        color="red",
        s=80,
        zorder=5,
        label=f"Selected epoch: {best_epoch}",
    )

    plt.title(args.title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean validation loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Create the output folder if it does not already exist.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)

    print(f"Best epoch: {best_epoch}")
    print(f"Lowest mean validation loss: {best_loss:.8f}")
    print(f"Graph saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()