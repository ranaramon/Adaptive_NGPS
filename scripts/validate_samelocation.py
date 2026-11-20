from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.triplet_dataset import VolumeTripletDataset
from datasets.volume_folder import list_volume_items
from ngps import build_nafnet
from ngps.config import NGPSConfig


def checkpoint_epoch(path: Path) -> int:
    """Extract the epoch number from model_E1.pth, model_E2.pth, etc."""
    match = re.fullmatch(r"model_E(\d+)\.pth", path.name)

    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")

    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a same-location model using noisy validation loss."
        )
    )

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--central-fraction",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device)

    checkpoints = sorted(
        checkpoint_dir.glob("model_E*.pth"),
        key=checkpoint_epoch,
    )

    if not checkpoints:
        raise RuntimeError(
            f"No model_E*.pth checkpoints found in {checkpoint_dir}"
        )

    # This loads subjects from data-root/val/noisy.
    validation_items = list_volume_items(args.data_root, "val")

    print("\nValidation subjects found:")
    for item in validation_items:
        print(f"  {item.case_id}")

    rows = []

    for checkpoint_path in checkpoints:
        epoch = checkpoint_epoch(checkpoint_path)

        print(f"\nValidating epoch {epoch}: {checkpoint_path.name}")

        state = torch.load(checkpoint_path, map_location=device)

        if "ngps_config" not in state:
            raise KeyError(
                f"No ngps_config found in {checkpoint_path.name}"
            )

        cfg_state = state["ngps_config"]

        if isinstance(cfg_state, dict):
            cfg = NGPSConfig(**cfg_state)
        else:
            cfg = cfg_state

        model = build_nafnet(cfg).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        subject_losses = []

        for item in validation_items:
            # Use full slices and no random crop for reproducible validation.
            dataset = VolumeTripletDataset(
                [item],
                cfg,
                crop_size=0,
                central_fraction=args.central_fraction,
            )

            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )

            slice_losses = []

            with torch.no_grad():
                for batch in tqdm(
                    loader,
                    desc=f"Subject {item.case_id}",
                    leave=False,
                ):
                    raw_center = batch["raw_center"].to(device)
                    raw_prev = batch["raw_prev_1"].to(device)
                    raw_next = batch["raw_next_1"].to(device)

                    prediction = model(raw_center)

                    loss_prev = F.mse_loss(prediction, raw_prev)
                    loss_next = F.mse_loss(prediction, raw_next)

                    loss = 0.5 * (loss_prev + loss_next)

                    slice_losses.append(float(loss.cpu()))

            if not slice_losses:
                raise RuntimeError(
                    f"No validation slices found for {item.case_id}"
                )

            subject_loss = float(np.mean(slice_losses))

            if not np.isfinite(subject_loss):
                raise ValueError(
                    f"Non-finite loss for {item.case_id}, epoch {epoch}"
                )

            subject_losses.append(subject_loss)

            rows.append(
                {
                    "epoch": epoch,
                    "subject": item.case_id,
                    "validation_loss": subject_loss,
                }
            )

            print(
                f"  {item.case_id}: "
                f"validation loss = {subject_loss:.8f}"
            )

        # Give each validation subject equal weight.
        mean_loss = float(np.mean(subject_losses))

        rows.append(
            {
                "epoch": epoch,
                "subject": "MEAN",
                "validation_loss": mean_loss,
            }
        )

        print(f"  Mean validation loss = {mean_loss:.8f}")

        del model

    mean_rows = [
        row for row in rows
        if row["subject"] == "MEAN"
    ]

    # If losses are equal, choose the earlier epoch.
    best = min(
        mean_rows,
        key=lambda row: (row["validation_loss"], row["epoch"]),
    )

    csv_path = checkpoint_dir / "validation_samelocation_loss.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "subject", "validation_loss"],
        )

        writer.writeheader()
        writer.writerows(rows)

    best_source = checkpoint_dir / f"model_E{best['epoch']}.pth"
    best_destination = checkpoint_dir / "model_best.pth"

    shutil.copy2(best_source, best_destination)

    print(f"\nResults saved to: {csv_path}")
    print(f"Best epoch: {best['epoch']}")
    print(f"Best mean validation loss: {best['validation_loss']:.8f}")
    print(f"Best checkpoint copied to: {best_destination}")


if __name__ == "__main__":
    main()