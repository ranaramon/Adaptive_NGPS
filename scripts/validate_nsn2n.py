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
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.triplet_dataset import VolumeTripletDataset
from datasets.volume_folder import list_volume_items
from ngps import (
    NSN2NLoss,
    build_nafnet,
    load_config,
    make_misalignment_mask,
)
from ngps.config import NGPSConfig


def checkpoint_epoch(path: Path) -> int:
    """Extract the epoch number from a name such as model_E3.pth."""
    match = re.fullmatch(r"model_E(\d+)\.pth", path.name)

    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")

    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select an NS-N2N checkpoint using noisy validation loss."
    )

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)

    parser.add_argument(
        "--config",
        default="configs/nsn2n_nafnet_default.yaml",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--central-fraction",
        type=float,
        default=1.0,
        help="Fraction of center slices to validate, matching training if needed.",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    # The config supplies the NS-N2N loss settings.
    config, _, raw_config = load_config(args.config)

    lambda_ic = float(
        raw_config.get("nsn2n", {}).get("lambda_ic", 1.0)
    )

    checkpoints = sorted(
        checkpoint_dir.glob("model_E*.pth"),
        key=checkpoint_epoch,
    )

    if not checkpoints:
        raise RuntimeError(
            f"No model_E*.pth checkpoints found in {checkpoint_dir}"
        )

    # Loads only the validation split's noisy volumes.
    validation_items = list_volume_items(args.data_root, "val")

    if not validation_items:
        raise RuntimeError("No validation volumes were found.")

    device = torch.device(args.device)
    rows = []

    for checkpoint_path in checkpoints:
        epoch = checkpoint_epoch(checkpoint_path)

        print(f"\nValidating epoch {epoch}: {checkpoint_path.name}")

        state = torch.load(checkpoint_path, map_location=device)

        # Use the model configuration stored in the checkpoint.
        cfg_state = state.get("ngps_config", {})

        if isinstance(cfg_state, dict):
            cfg = NGPSConfig(**cfg_state)
        else:
            cfg = cfg_state

        model = build_nafnet(cfg).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        # Match the NS-N2N training objective.
        criterion = NSN2NLoss(
            lambda_rc=cfg.rc_weight,
            lambda_ic=lambda_ic,
        )

        subject_losses = []

        for item in validation_items:
            # One subject at a time so that subjects get equal weight.
            #
            # crop_size=0 uses full slices and avoids random validation crops.
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

            batch_losses = []

            with torch.no_grad():
                for batch in tqdm(
                    loader,
                    desc=f"Subject {item.case_id}",
                    leave=False,
                ):
                    raw_center = batch["raw_center"].to(device)
                    raw_prev = batch["raw_prev_1"].to(device)
                    raw_next = batch["raw_next_1"].to(device)

                    guide_center = batch["guide_center"].to(device)
                    guide_prev = batch["guide_prev_1"].to(device)
                    guide_next = batch["guide_next_1"].to(device)

                    mask_prev = make_misalignment_mask(
                        guide_center,
                        guide_prev,
                        cfg.mask_threshold,
                    )

                    mask_next = make_misalignment_mask(
                        guide_center,
                        guide_next,
                        cfg.mask_threshold,
                    )

                    loss_out = criterion(
                        model,
                        raw_center,
                        raw_prev,
                        raw_next,
                        mask_prev,
                        mask_next,
                    )

                    batch_losses.append(
                        float(loss_out.total.detach().cpu())
                    )

            subject_loss = float(np.mean(batch_losses))

            if not np.isfinite(subject_loss):
                raise ValueError(
                    f"Non-finite validation loss for {item.case_id}, "
                    f"epoch {epoch}"
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

    # If losses are exactly equal, select the earlier epoch.
    best = min(
        mean_rows,
        key=lambda row: (row["validation_loss"], row["epoch"]),
    )

    csv_path = checkpoint_dir / "validation_nsn2n_loss.csv"

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
    print(
        "Best mean validation loss: "
        f"{best['validation_loss']:.8f}"
    )
    print(f"Best checkpoint copied to: {best_destination}")


if __name__ == "__main__":
    main()