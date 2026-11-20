from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.triplet_dataset import VolumeTripletDataset
from datasets.volume_folder import list_volume_items

from ngps import (
    NGPSHybridLoss,
    NeighborGuidedPatchSampler,
    build_nafnet,
    load_config,
)

from ngps.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train adaptive NGPS, optionally continuing from a checkpoint."
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the dataset for the selected noise level.",
    )

    parser.add_argument(
        "--config",
        default="configs/ngps_nafnet_default.yaml",
        help="Path to the training configuration.",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/ngps",
        help="Directory for new checkpoints.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cuda or cpu.",
    )

    parser.add_argument(
        "--central-fraction",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a checkpoint after this many epochs.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a saved model_E<number>.pth checkpoint.",
    )

    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Completed epoch number represented by the checkpoint.",
    )

    args = parser.parse_args()

    if args.save_every < 1:
        parser.error("--save-every must be at least 1.")

    if args.resume and args.resume_epoch is None:
        parser.error("--resume requires --resume-epoch.")

    if args.resume_epoch is not None and not args.resume:
        parser.error("--resume-epoch requires --resume.")

    if args.resume_epoch is not None and args.resume_epoch < 1:
        parser.error("--resume-epoch must be at least 1.")

    # Load the requested configuration.
    cfg, train_cfg, _ = load_config(args.config)

    seed_everything(train_cfg.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Dataset and data loader
    # ---------------------------------------------------------
    items = list_volume_items(args.data_root, "train")

    dataset = VolumeTripletDataset(
        items,
        cfg,
        crop_size=train_cfg.crop_size,
        central_fraction=args.central_fraction,
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )

    # ---------------------------------------------------------
    # Model, sampler, loss, and optimizer
    # ---------------------------------------------------------
    model = build_nafnet(cfg).to(device)

    sampler = NeighborGuidedPatchSampler(cfg).to(device)
    sampler.eval()

    criterion = NGPSHybridLoss(
        lambda_rc=cfg.rc_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    # ---------------------------------------------------------
    # Optional checkpoint continuation
    # ---------------------------------------------------------
    start_epoch = 1

    if args.resume:
        checkpoint_path = Path(args.resume)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        if "model" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain model weights."
            )

        saved_config = checkpoint.get("ngps_config")

        if saved_config is None:
            raise KeyError(
                "Checkpoint does not contain ngps_config."
            )

        if saved_config != asdict(cfg):
            raise ValueError(
                "The current NGPS configuration differs from the "
                "checkpoint configuration. Use the configuration "
                "that was used for this checkpoint."
            )

        saved_epoch = checkpoint.get("epoch")

        if (
            saved_epoch is not None
            and saved_epoch != args.resume_epoch
        ):
            raise ValueError(
                f"Checkpoint records epoch {saved_epoch}, but "
                f"--resume-epoch is {args.resume_epoch}."
            )

        start_epoch = args.resume_epoch + 1

        if start_epoch > train_cfg.epochs:
            raise ValueError(
                f"The checkpoint is already at epoch "
                f"{args.resume_epoch}, while the configuration "
                f"specifies {train_cfg.epochs} total epochs."
            )

        model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer state restored.")
        else:
            print(
                "Optimizer state was not saved in this checkpoint. "
                "The optimizer will start fresh."
            )

        saved_training_config = checkpoint.get("training_config")

        if saved_training_config != asdict(train_cfg):
            print(
                "Note: Current training settings differ from the "
                "settings saved in the checkpoint. Check batch size, "
                "crop size, learning rate, and total epochs."
            )

        print(f"Loaded model weights from: {checkpoint_path}")
        print(f"Continuing from epoch {start_epoch}.")
        print(
            "Random-number and data-loader states are not restored."
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        model.train()
        losses = []

        loop = tqdm(
            loader,
            desc=f"NGPS epoch {epoch}/{train_cfg.epochs}",
        )

        for batch in loop:
            # Center slice and its filtered guide.
            raw_center = batch["raw_center"].to(device)
            guide_center = batch["guide_center"].to(device)

            # Immediate neighbors: z-1 and z+1.
            raw_prev_1 = batch["raw_prev_1"].to(device)
            raw_next_1 = batch["raw_next_1"].to(device)

            guide_prev_1 = batch["guide_prev_1"].to(device)
            guide_next_1 = batch["guide_next_1"].to(device)

            # Farther neighbors: z-2 and z+2.
            raw_prev_2 = batch["raw_prev_2"].to(device)
            raw_next_2 = batch["raw_next_2"].to(device)

            guide_prev_2 = batch["guide_prev_2"].to(device)
            guide_next_2 = batch["guide_next_2"].to(device)

            batch_size = raw_center.shape[0]

            # Immediate neighbors exist for sampled center slices.
            valid_immediate = torch.ones(
                (batch_size, 1, 1, 1),
                dtype=torch.bool,
                device=device,
            )

            # Farther neighbors may be unavailable near boundaries.
            valid_prev_2 = (
                batch["valid_prev_2"]
                .to(device=device, dtype=torch.bool)
                .reshape(batch_size, 1, 1, 1)
            )

            valid_next_2 = (
                batch["valid_next_2"]
                .to(device=device, dtype=torch.bool)
                .reshape(batch_size, 1, 1, 1)
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

            # Radius 2 adds the slices two positions away.
            if cfg.slice_search_radius >= 2:
                raw_candidates.extend([
                    raw_prev_2,
                    raw_next_2,
                ])

                guide_candidates.extend([
                    guide_prev_2,
                    guide_next_2,
                ])

                valid_candidates.extend([
                    valid_prev_2,
                    valid_next_2,
                ])

            # Build candidate-specific targets and masks.
            with torch.no_grad():
                targets = sampler(
                    guide_center=guide_center,
                    guide_candidates=guide_candidates,
                    raw_candidates=raw_candidates,
                    valid_candidates=valid_candidates,
                )

            # Denoise center and candidate slices.
            out_center = model(raw_center)

            out_candidates = [
                model(raw_neighbor)
                for raw_neighbor in raw_candidates
            ]

            # Rejected candidates contribute no loss at that location.
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

            optimizer.zero_grad(set_to_none=True)
            loss_out.total.backward()
            optimizer.step()

            losses.append(
                float(loss_out.total.detach().cpu())
            )

            # Count accepted candidates at each location.
            match_count = torch.stack(
                [
                    candidate.accepted
                    for candidate in targets.candidates
                ],
                dim=0,
            ).sum(dim=0)

            fraction_no_match = (
                (match_count == 0)
                .float()
                .mean()
                .item()
            )

            loop.set_postfix(
                loss=f"{sum(losses) / len(losses):.4f}",
                no_match=f"{fraction_no_match:.1%}",
            )

        # Save a checkpoint after completed epochs.
        if epoch % args.save_every == 0 or epoch == train_cfg.epochs:
            checkpoint_path = out_dir / f"model_E{epoch}.pth"

            if checkpoint_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite {checkpoint_path}. "
                    "Choose a new --output-dir."
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ngps_config": asdict(cfg),
                    "training_config": asdict(train_cfg),
                },
                checkpoint_path,
            )

            print(f"Saved checkpoint to {checkpoint_path}")

    # Save the final checkpoint.
    final_path = out_dir / "model_final.pth"

    if final_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {final_path}. "
            "Choose a new --output-dir."
        )

    torch.save(
        {
            "epoch": train_cfg.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ngps_config": asdict(cfg),
            "training_config": asdict(train_cfg),
        },
        final_path,
    )

    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()