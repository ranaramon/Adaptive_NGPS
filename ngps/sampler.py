from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NGPSConfig
from .guide import make_misalignment_mask

@dataclass
class CandidateTarget:
    """Results for one neighboring slice."""

    target: torch.Tensor
    dynamic_mask: torch.Tensor
    accepted: torch.Tensor


@dataclass
class NGPSTargets:
    """Results for all candidate slices."""

    candidates: list[CandidateTarget]

# @dataclass
# class NGPSTargets:
#     target_prev: torch.Tensor
#     target_next: torch.Tensor
#
#     # Existing direction-aware dynamic masks.
#     mask_prev: torch.Tensor
#     mask_next: torch.Tensor
#
#     # New: indicates whether at least one slice passed the
#     # matching threshold at each location.
#     has_match_prev: torch.Tensor
#     has_match_next: torch.Tensor
#
#     # New: number of accepted slice candidates at each location.
#     match_count_prev: torch.Tensor
#     match_count_next: torch.Tensor



class NeighborGuidedPatchSampler(nn.Module):
    """Modification: Build candidate-specific patch targets, masks, and match decisions."""
    """Dense local patch retrieval across multiple neighboring slices.

    For each candidate slice, the sampler performs the original Top-K
    spatial patch search. A slice is accepted at a center location if its
    mean normalized Top-K cost is below the configured threshold.

    Accepted slice targets are averaged separately for the previous and
    next directions. If no candidate is accepted in one direction, its
    target is a placeholder and its has_match mask is False.
    """

    def __init__(self, cfg: NGPSConfig = NGPSConfig()) -> None:
        super().__init__()

        self.cfg = cfg
        self.pad = cfg.search_window // 2
        self.num_shifts = cfg.search_window * cfg.search_window

        kernel = torch.ones(
            (
                self.num_shifts,
                1,
                cfg.patch_size,
                cfg.patch_size,
            ),
            dtype=torch.float32,
        )

        self.register_buffer(
            "ssd_kernel",
            kernel,
            persistent=False,
        )

    @staticmethod
    def _validate_image(x: torch.Tensor, name: str) -> None:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape (B, 1, H, W)."
            )

    def _shift_stack(self, image: torch.Tensor) -> torch.Tensor:
        """Stack all local-window shifts along the channel dimension."""

        self._validate_image(image, "image")

        _, _, height, width = image.shape

        padded = F.pad(
            image,
            (self.pad, self.pad, self.pad, self.pad),
            mode="reflect",
        )

        shifts = []

        for dy in range(self.cfg.search_window):
            for dx in range(self.cfg.search_window):
                shifts.append(
                    padded[
                        :,
                        :,
                        dy : dy + height,
                        dx : dx + width,
                    ]
                )

        return torch.cat(shifts, dim=1)

    def _ssd_stack(
        self,
        guide_center: torch.Tensor,
        guide_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        """Return the patch SSD for every spatial shift."""

        shifted_guides = self._shift_stack(guide_neighbor)
        diff_sq = (guide_center - shifted_guides).pow(2)

        kernel = self.ssd_kernel.to(
            device=diff_sq.device,
            dtype=diff_sq.dtype,
        )

        return F.conv2d(
            diff_sq,
            kernel,
            padding=self.cfg.patch_size // 2,
            groups=self.num_shifts,
        )

    @torch.no_grad()
    def forward_direction(
        self,
        guide_center: torch.Tensor,
        guide_neighbor: torch.Tensor,
        raw_neighbor: torch.Tensor,
        return_cost: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Perform the original Top-K search in one neighboring slice.

        If return_cost is True, return the mean Top-K SSD divided by
        patch area along with the target.
        """

        ssd = self._ssd_stack(guide_center, guide_neighbor)

        top_values, top_indices = torch.topk(
            ssd,
            k=self.cfg.topk,
            dim=1,
            largest=False,
        )

        shifted_raw = self._shift_stack(raw_neighbor)

        top_raw = torch.gather(
            shifted_raw,
            dim=1,
            index=top_indices,
        )

        target = top_raw.mean(dim=1, keepdim=True).detach()

        if not return_cost:
            return target

        cost = (
            top_values.mean(dim=1, keepdim=True)
            / float(self.cfg.patch_size ** 2)
        )

        return target, cost.detach()

    # @torch.no_grad()
    # def forward_candidates(
    #     self,
    #     guide_center: torch.Tensor,
    #     guide_candidates: Sequence[torch.Tensor],
    #     raw_candidates: Sequence[torch.Tensor],
    #     valid_candidates: Sequence[torch.Tensor] | None = None,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Combine accepted candidate slices for one direction.
    #
    #     Args:
    #         guide_center:
    #             Center guide with shape (B, 1, H, W).
    #
    #         guide_candidates:
    #             Candidate guides on one side of the center slice.
    #             Each tensor has shape (B, 1, H, W).
    #
    #         raw_candidates:
    #             Raw slices corresponding to guide_candidates, in the
    #             same order. Each has shape (B, 1, H, W).
    #
    #         valid_candidates:
    #             Optional per-candidate Boolean tensors of shape
    #             (B, 1, 1, 1). False marks a slice that is unavailable
    #             for a particular batch item, such as at a boundary.
    #
    #             Invalid entries still need placeholder image tensors
    #             with valid shapes because the search is computed before
    #             the validity mask is applied.
    #
    #     Returns:
    #         target:
    #             Mean of accepted slice targets, shape (B, 1, H, W).
    #             Zero is used as a placeholder where no slice matches.
    #
    #         has_match:
    #             Boolean tensor of shape (B, 1, H, W).
    #
    #         match_count:
    #             Number of accepted slice candidates at each location,
    #             shape (B, 1, H, W).
    #     """
    #
    #     self._validate_image(guide_center, "guide_center")
    #
    #     if len(guide_candidates) != len(raw_candidates):
    #         raise ValueError(
    #             "guide_candidates and raw_candidates must have "
    #             "the same length."
    #         )
    #
    #     if valid_candidates is not None:
    #         if len(valid_candidates) != len(guide_candidates):
    #             raise ValueError(
    #                 "valid_candidates must have one entry per candidate."
    #             )
    #
    #     # No candidate slices available on this side.
    #     if len(guide_candidates) == 0:
    #         target = torch.zeros_like(guide_center)
    #         has_match = torch.zeros_like(
    #             guide_center,
    #             dtype=torch.bool,
    #         )
    #         match_count = torch.zeros_like(
    #             guide_center,
    #             dtype=torch.long,
    #         )
    #         return target, has_match, match_count
    #
    #     slice_targets = []
    #     slice_matches = []
    #
    #     for index, (guide_neighbor, raw_neighbor) in enumerate(
    #         zip(guide_candidates, raw_candidates)
    #     ):
    #         self._validate_image(
    #             guide_neighbor,
    #             f"guide_candidates[{index}]",
    #         )
    #         self._validate_image(
    #             raw_neighbor,
    #             f"raw_candidates[{index}]",
    #         )
    #
    #         if guide_neighbor.shape != guide_center.shape:
    #             raise ValueError(
    #                 f"guide_candidates[{index}] must have the "
    #                 "same shape as guide_center."
    #             )
    #
    #         if raw_neighbor.shape != guide_center.shape:
    #             raise ValueError(
    #                 f"raw_candidates[{index}] must have the "
    #                 "same shape as guide_center."
    #             )
    #
    #         target, cost = self.forward_direction(
    #             guide_center,
    #             guide_neighbor,
    #             raw_neighbor,
    #             return_cost=True,
    #         )
    #
    #         # A candidate slice passes only where its cost is low enough.
    #         accepted = cost <= self.cfg.match_threshold
    #
    #         if valid_candidates is not None:
    #             valid = valid_candidates[index]
    #
    #             expected_shape = (
    #                 guide_center.shape[0],
    #                 1,
    #                 1,
    #                 1,
    #             )
    #
    #             if valid.shape != expected_shape:
    #                 raise ValueError(
    #                     f"valid_candidates[{index}] must have "
    #                     f"shape {expected_shape}."
    #                 )
    #
    #             valid = valid.to(
    #                 device=guide_center.device,
    #                 dtype=torch.bool,
    #             )
    #
    #             accepted = accepted & valid
    #
    #         slice_targets.append(target)
    #         slice_matches.append(accepted)
    #
    #     # Each input has shape (B, 1, H, W).
    #     # After stacking, shape is (B, number_of_slices, H, W).
    #     targets = torch.cat(slice_targets, dim=1)
    #     matches = torch.cat(slice_matches, dim=1)
    #
    #     weights = matches.to(dtype=targets.dtype)
    #
    #     match_count = matches.sum(dim=1, keepdim=True)
    #     has_match = match_count > 0
    #
    #     target_sum = (targets * weights).sum(dim=1, keepdim=True)
    #
    #     # Clamp avoids division by zero. Where there is no match,
    #     # target_sum is zero and target is only a placeholder.
    #     target = target_sum / match_count.clamp_min(1).to(targets.dtype)
    #
    #     return target.detach(), has_match.detach(), match_count.detach()
    #
    # @torch.no_grad()
    # def forward(
    #     self,
    #     guide_center: torch.Tensor,
    #
    #     # Immediate neighbors are still needed to construct the
    #     # original direction-aware masks.
    #     guide_prev: torch.Tensor,
    #     guide_next: torch.Tensor,
    #
    #     # Candidate guides and raw slices for each direction.
    #     guide_prev_candidates: Sequence[torch.Tensor],
    #     raw_prev_candidates: Sequence[torch.Tensor],
    #     guide_next_candidates: Sequence[torch.Tensor],
    #     raw_next_candidates: Sequence[torch.Tensor],
    #
    #     # Optional validity flags for volume boundaries.
    #     valid_prev_candidates: Sequence[torch.Tensor] | None = None,
    #     valid_next_candidates: Sequence[torch.Tensor] | None = None,
    # ) -> NGPSTargets:
    #
    #     (
    #         target_prev,
    #         has_match_prev,
    #         match_count_prev,
    #     ) = self.forward_candidates(
    #         guide_center,
    #         guide_prev_candidates,
    #         raw_prev_candidates,
    #         valid_prev_candidates,
    #     )
    #
    #     (
    #         target_next,
    #         has_match_next,
    #         match_count_next,
    #     ) = self.forward_candidates(
    #         guide_center,
    #         guide_next_candidates,
    #         raw_next_candidates,
    #         valid_next_candidates,
    #     )
    #
    #     # Preserve the original mask behavior for the initial experiment.
    #     mask_prev = make_misalignment_mask(
    #         guide_center,
    #         guide_prev,
    #         self.cfg.mask_threshold,
    #     )
    #
    #     mask_next = make_misalignment_mask(
    #         guide_center,
    #         guide_next,
    #         self.cfg.mask_threshold,
    #     )
    #
    #     return NGPSTargets(
    #         target_prev=target_prev,
    #         target_next=target_next,
    #         mask_prev=mask_prev.detach(),
    #         mask_next=mask_next.detach(),
    #         has_match_prev=has_match_prev,
    #         has_match_next=has_match_next,
    #         match_count_prev=match_count_prev,
    #         match_count_next=match_count_next,
    #     )
    @torch.no_grad()
    def forward(
        self,
        guide_center: torch.Tensor,
        guide_candidates: Sequence[torch.Tensor],
        raw_candidates: Sequence[torch.Tensor],
        valid_candidates: Sequence[torch.Tensor],
    ) -> NGPSTargets:
        """Build a target, mask, and acceptance map for each candidate.

        Image tensors have shape (B, 1, H, W). Each validity tensor
        has shape (B, 1, 1, 1), with False for unavailable slices.
        """

        self._validate_image(guide_center, "guide_center")

        if not (
            len(guide_candidates)
            == len(raw_candidates)
            == len(valid_candidates)
        ):
            raise ValueError(
                "Guide, raw, and validity lists must have the same length."
            )

        candidate_results = []
        batch_size = guide_center.shape[0]

        for index, (guide_neighbor, raw_neighbor, valid) in enumerate(
            zip(guide_candidates, raw_candidates, valid_candidates)
        ):
            self._validate_image(
                guide_neighbor,
                f"guide_candidates[{index}]",
            )
            self._validate_image(
                raw_neighbor,
                f"raw_candidates[{index}]",
            )

            if guide_neighbor.shape != guide_center.shape:
                raise ValueError(
                    f"guide_candidates[{index}] must have the "
                    "same shape as guide_center."
                )

            if raw_neighbor.shape != guide_center.shape:
                raise ValueError(
                    f"raw_candidates[{index}] must have the "
                    "same shape as guide_center."
                )

            expected_valid_shape = (batch_size, 1, 1, 1)

            if valid.shape != expected_valid_shape:
                raise ValueError(
                    f"valid_candidates[{index}] must have "
                    f"shape {expected_valid_shape}."
                )

            valid = valid.to(
                device=guide_center.device,
                dtype=torch.bool,
            )

            # Use the existing Top-K spatial search in this slice.
            target, cost = self.forward_direction(
                guide_center,
                guide_neighbor,
                raw_neighbor,
                return_cost=True,
            )

            # Compare the center and this candidate at the same position.
            dynamic_mask = make_misalignment_mask(
                guide_center,
                guide_neighbor,
                self.cfg.mask_threshold,
            )

            # A candidate is accepted only if it exists and its
            # normalized mean Top-K cost passes the threshold.
            accepted = valid & (cost <= self.cfg.match_threshold)

            candidate_results.append(
                CandidateTarget(
                    target=target.detach(),
                    dynamic_mask=dynamic_mask.detach(),
                    accepted=accepted.detach(),
                )
            )

        return NGPSTargets(candidates=candidate_results)