import torch

from ngps.losses import NGPSHybridLoss


def main():
    criterion = NGPSHybridLoss(lambda_rc=0.5)

    # Pretend this is the network's prediction for the center slice.
    out_center = torch.full(
        (1, 1, 4, 4),
        0.5,
        requires_grad=True,
    )

    # Four deliberately different neighboring candidates.
    raw_candidates = [
        torch.full_like(out_center, value)
        for value in (0.1, 0.3, 0.7, 0.9)
    ]

    out_candidates = [
        torch.full_like(out_center, value)
        for value in (0.2, 0.4, 0.6, 0.8)
    ]

    candidate_targets = [
        torch.full_like(out_center, value)
        for value in (0.15, 0.35, 0.65, 0.85)
    ]

    # Mix static and dynamic masks.
    dynamic_masks = [
        torch.zeros_like(out_center),
        torch.ones_like(out_center),
        torch.zeros_like(out_center),
        torch.ones_like(out_center),
    ]

    # Reject every candidate at every location.
    accepted_masks = [
        torch.zeros_like(out_center, dtype=torch.bool)
        for _ in range(4)
    ]

    result = criterion(
        out_center=out_center,
        raw_candidates=raw_candidates,
        out_candidates=out_candidates,
        candidate_targets=candidate_targets,
        dynamic_masks=dynamic_masks,
        accepted_masks=accepted_masks,
    )

    result.total.backward()

    print("N2N loss:    ", result.n2n.item())
    print("Dynamic loss:", result.dynamic.item())
    print("RC loss:     ", result.rc.item())
    print("Total loss:  ", result.total.item())
    print("Max gradient:", out_center.grad.abs().max().item())

    assert torch.isfinite(result.total)
    assert result.n2n.item() == 0.0
    assert result.dynamic.item() == 0.0
    assert result.rc.item() == 0.0
    assert result.total.item() == 0.0
    assert torch.count_nonzero(out_center.grad).item() == 0

    print("\nPASS: rejected candidates contributed no loss or gradient.")


if __name__ == "__main__":
    main()