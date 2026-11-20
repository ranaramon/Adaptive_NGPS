import torch

from ngps.config import NGPSConfig
from ngps.sampler import NeighborGuidedPatchSampler


def main():
    # Small settings make this test quick.
    cfg = NGPSConfig(
        patch_size=3,
        search_window=3,
        topk=1,
        match_threshold=0.01,
    )

    sampler = NeighborGuidedPatchSampler(cfg).eval()

    # Center guide and two candidate guides.
    center = torch.zeros(1, 1, 8, 8)

    matching_guide = torch.zeros_like(center)
    different_guide = torch.ones_like(center)

    # Different raw values make the targets easy to identify.
    matching_raw = torch.full_like(center, 0.25)
    different_raw = torch.full_like(center, 0.75)

    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    invalid = torch.zeros(1, 1, 1, 1, dtype=torch.bool)

    with torch.no_grad():
        results = sampler(
            guide_center=center,
            guide_candidates=[
                matching_guide,
                different_guide,
                matching_guide,
            ],
            raw_candidates=[
                matching_raw,
                different_raw,
                matching_raw,
            ],
            valid_candidates=[
                valid,    # Valid and similar: accept.
                valid,    # Valid but different: reject.
                invalid,  # Similar but unavailable: reject.
            ],
        )

    accepted_match = results.candidates[0].accepted
    accepted_different = results.candidates[1].accepted
    accepted_invalid = results.candidates[2].accepted

    assert accepted_match.all(), "Matching candidate should be accepted."
    assert not accepted_different.any(), "Different candidate should be rejected."
    assert not accepted_invalid.any(), "Invalid candidate should be rejected."

    assert torch.allclose(
        results.candidates[0].target,
        torch.full_like(center, 0.25),
    ), "The accepted candidate should retrieve its raw value."

    print("PASS: similar candidate accepted.")
    print("PASS: dissimilar candidate rejected.")
    print("PASS: unavailable candidate rejected.")
    print("PASS: accepted target retrieved from the raw candidate.")


if __name__ == "__main__":
    main()