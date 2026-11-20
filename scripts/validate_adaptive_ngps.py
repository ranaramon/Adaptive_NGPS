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
from ngps import NGPSHybridLoss, NeighborGuidedPatchSampler, build_nafnet
from ngps.config import NGPSConfig


def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r"model_E(\d+)\.pth", path.name)

    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")

    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select an adaptive NGPS checkpoint using validation loss."
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

    validation_items = list_volume_items(args.data_root, "val")

    if not validation_items:
        raise RuntimeError("No validation volumes were found.")

    rows = []
    radii_seen = set()

    for checkpoint_path in checkpoints:
        epoch = checkpoint_epoch(checkpoint_path)
        print(f"\nValidating epoch {epoch}: {checkpoint_path.name}")

        state = torch.load(checkpoint_path, map_location=device)

        if "ngps_config" not in state:
            raise KeyError(
                f"{checkpoint_path.name} has no saved ngps_config."
            )

        cfg_state = state["ngps_config"]

        if isinstance(cfg_state, dict):
            cfg = NGPSConfig(**cfg_state)
        else:
            cfg = cfg_state

        radii_seen.add(cfg.slice_search_radius)

        if len(radii_seen) > 1:
            raise ValueError(
                "The checkpoint folder contains different slice radii. "
                "Validate each radius in a separate folder."
            )

        model = build_nafnet(cfg).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        sampler = NeighborGuidedPatchSampler(cfg).to(device)
        sampler.eval()

        criterion = NGPSHybridLoss(lambda_rc=cfg.rc_weight)

        subject_losses = []

        for item in validation_items:
            # A separate dataset per subject allows equal subject weighting.
            # crop_size=0 avoids random validation crops.
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
                    guide_center = batch["guide_center"].to(device)

                    raw_prev_1 = batch["raw_prev_1"].to(device)
                    raw_next_1 = batch["raw_next_1"].to(device)

                    guide_prev_1 = batch["guide_prev_1"].to(device)
                    guide_next_1 = batch["guide_next_1"].to(device)

                    batch_size = raw_center.shape[0]

                    valid_immediate = torch.ones(
                        (batch_size, 1, 1, 1),
                        dtype=torch.bool,
                        device=device,
                    )

                    raw_candidates = [
                        raw_prev_1,
                        raw_next_1,
                    ]

                    guide_candidates = [
                        guide_prev_1,
                        guide_next_1,
                    ]

                    valid_candidates = [
                        valid_immediate,
                        valid_immediate,
                    ]

                    if cfg.slice_search_radius >= 2:
                        raw_prev_2 = batch["raw_prev_2"].to(device)
                        raw_next_2 = batch["raw_next_2"].to(device)

                        guide_prev_2 = batch["guide_prev_2"].to(device)
                        guide_next_2 = batch["guide_next_2"].to(device)

                        valid_prev_2 = batch["valid_prev_2"].to(
                            device=device,
                            dtype=torch.bool,
                        ).reshape(batch_size, 1, 1, 1)

                        valid_next_2 = batch["valid_next_2"].to(
                            device=device,
                            dtype=torch.bool,
                        ).reshape(batch_size, 1, 1, 1)

                        raw_candidates.extend(
                            [raw_prev_2, raw_next_2]
                        )

                        guide_candidates.extend(
                            [guide_prev_2, guide_next_2]
                        )

                        valid_candidates.extend(
                            [valid_prev_2, valid_next_2]
                        )

                    targets = sampler(
                        guide_center=guide_center,
                        guide_candidates=guide_candidates,
                        raw_candidates=raw_candidates,
                        valid_candidates=valid_candidates,
                    )

                    out_center = model(raw_center)

                    out_candidates = [
                        model(raw_neighbor)
                        for raw_neighbor in raw_candidates
                    ]

                    loss_out = criterion(
                        out_center=out_center,
                        raw_candidates=raw_candidates,
                        out_candidates=out_candidates,
                        candidate_targets=[
                            candidate.target
                            for candidate in targets.candidates
                        ],
                        dynamic_masks=[
                            candidate.dynamic_mask
                            for candidate in targets.candidates
                        ],
                        accepted_masks=[
                            candidate.accepted
                            for candidate in targets.candidates
                        ],
                    )

                    slice_losses.append(
                        float(loss_out.total.detach().cpu())
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
                    "radius": cfg.slice_search_radius,
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
                "radius": cfg.slice_search_radius,
                "subject": "MEAN",
                "validation_loss": mean_loss,
            }
        )

        print(f"  Mean validation loss = {mean_loss:.8f}")

        del model, sampler

    mean_rows = [
        row for row in rows
        if row["subject"] == "MEAN"
    ]

    # If losses are exactly equal, select the earlier epoch.
    best = min(
        mean_rows,
        key=lambda row: (
            row["validation_loss"],
            row["epoch"],
        ),
    )

    csv_path = checkpoint_dir / "validation_adaptive_loss.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "radius",
                "subject",
                "validation_loss",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    best_source = checkpoint_dir / f"model_E{best['epoch']}.pth"
    best_destination = checkpoint_dir / "model_best.pth"

    shutil.copy2(best_source, best_destination)

    print(f"\nResults saved to: {csv_path}")
    print(f"Radius: {best['radius']}")
    print(f"Best epoch: {best['epoch']}")
    print(f"Best validation loss: {best['validation_loss']:.8f}")
    print(f"Best checkpoint copied to: {best_destination}")


if __name__ == "__main__":
    main()