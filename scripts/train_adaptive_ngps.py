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
        description="Train NGPS on local .npy volumes."
    )

    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/ngps_nafnet_default.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/ngps",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
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
    )

    args = parser.parse_args()

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
    # Training
    # ---------------------------------------------------------
    for epoch in range(1, train_cfg.epochs + 1):
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

            # Immediate neighboring slices: z−1 and z+1.
            raw_prev_1 = batch["raw_prev_1"].to(device)
            raw_next_1 = batch["raw_next_1"].to(device)

            guide_prev_1 = batch["guide_prev_1"].to(device)
            guide_next_1 = batch["guide_next_1"].to(device)

            # Farther neighboring slices: z−2 and z+2.
            raw_prev_2 = batch["raw_prev_2"].to(device)
            raw_next_2 = batch["raw_next_2"].to(device)

            guide_prev_2 = batch["guide_prev_2"].to(device)
            guide_next_2 = batch["guide_next_2"].to(device)

            batch_size = raw_center.shape[0]

            # The dataset uses margin=1, so both immediate
            # neighbors exist for every sampled center slice.
            valid_immediate = torch.ones(
                (batch_size, 1, 1, 1),
                dtype=torch.bool,
                device=device,
            )

            # Farther neighbors may not exist near volume boundaries.
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

            # Candidate order starts as:
            # 0: z−1
            # 1: z+1
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

            # With radius 2, add:
            # 2: z−2
            # 3: z+2
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

            # -------------------------------------------------
            # Build a target, mask, and acceptance map for
            # each candidate slice.
            # -------------------------------------------------
            with torch.no_grad():
                targets = sampler(
                    guide_center=guide_center,
                    guide_candidates=guide_candidates,
                    raw_candidates=raw_candidates,
                    valid_candidates=valid_candidates,
                )

            # -------------------------------------------------
            # Denoise the center and each candidate slice.
            # Candidate outputs are needed for regional
            # consistency.
            # -------------------------------------------------
            out_center = model(raw_center)

            out_candidates = [
                model(raw_neighbor)
                for raw_neighbor in raw_candidates
            ]

            # -------------------------------------------------
            # Calculate candidate-specific loss.
            # Rejected candidates do not contribute at the
            # corresponding locations.
            # -------------------------------------------------
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

            # Count accepted candidates at each center location.
            # With radius 2, values range from 0 to 4.
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

        # Save periodic checkpoints inside the epoch loop.
        if epoch % args.save_every == 0 or epoch == train_cfg.epochs:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ngps_config": asdict(cfg),
                    "training_config": asdict(train_cfg),
                },
                out_dir / f"model_E{epoch}.pth",
            )

    # Save the final checkpoint after all epochs.
    torch.save(
        {
            "model": model.state_dict(),
            "ngps_config": asdict(cfg),
            "training_config": asdict(train_cfg),
        },
        out_dir / "model_final.pth",
    )

    print(f"Saved checkpoint to {out_dir / 'model_final.pth'}")


if __name__ == "__main__":
    main()