"""neutral_mdlm_loss.py

Structured charge-neutral training objective for MatterGen's continuous-time
MDLM.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
from neutral_layer.generation.dp import compute_q_max, flat_to_padded, neutral_marginals_diff

from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.corruption.candidate_pinning import pin_committed_candidates
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.mdlm.schedule import mdlm_loss_weight


def _validate_loss_configuration(
    *,
    score_model_output: torch.Tensor,
    reduce: str,
    charge_of: list[int],
    mask_idx: int,
) -> None:
    """Validate the absorbing-MDLM vocabulary and loss options."""
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
    if score_model_output.ndim != 2:
        raise ValueError(
            "score_model_output must have shape [N_atoms, K], got "
            f"{score_model_output.shape}"
        )
    K = score_model_output.shape[-1]
    if K < 1:
        raise ValueError("score_model_output must contain at least the MASK class")
    if len(charge_of) != K:
        raise ValueError(
            f"charge_of has length {len(charge_of)}, but logits have {K} classes"
        )
    if mask_idx != K - 1:
        raise ValueError(
            "absorbing MDLM requires MASK to be the final class; "
            f"got mask_idx={mask_idx}, K={K}"
        )
    if charge_of[mask_idx] != 0:
        raise ValueError("the absorbing MASK class must have charge zero")


def _validate_neutral_targets(
    x0_pad: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    charge_tensor: torch.LongTensor,
) -> None:
    """Fail loudly when a clean training target is outside the constraint set."""
    K = charge_tensor.shape[0]
    real_x0 = x0_pad[attention_mask]
    if bool(((real_x0 < 0) | (real_x0 >= K)).any()):
        raise ValueError(f"clean species indices must lie in [0, {K - 1}]")

    target_charge = charge_tensor[x0_pad.clamp(0, K - 1)]
    target_charge = torch.where(attention_mask, target_charge, torch.zeros_like(target_charge))
    total_charge = target_charge.sum(dim=1)
    if bool((total_charge != 0).any()):
        bad = (total_charge != 0).nonzero(as_tuple=True)[0].tolist()
        totals = {b: int(total_charge[b].item()) for b in bad}
        raise ValueError(
            "neutral_mdlm_loss requires charge-neutral clean targets; "
            f"invalid structure charge sums: {totals}"
        )


def _require_finite_partition(log_z: torch.Tensor, *, t: torch.Tensor) -> None:
    """Raise with batch context instead of propagating a non-finite loss."""
    finite = torch.isfinite(log_z)
    if bool(finite.all()):
        return
    bad = (~finite).nonzero(as_tuple=True)[0].tolist()
    times = {b: float(t[b].item()) for b in bad}
    raise RuntimeError(
        f"neutral_mdlm_loss has non-finite constrained partition for structures {bad}; "
        f"t={times}. Check the species vocabulary, pinned assignments, and model logits."
    )


def neutral_mdlm_loss(
    *,
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    charge_of: list[int],
    mask_idx: int,
    **_,
) -> torch.Tensor:
    """Structured charge-neutral MDLM loss for atomic numbers.

    Replaces the raw per-site clean-state probability at each masked site
    with its exact charge-neutral one-site marginal, obtained from a single
    forward-backward DP pass over the whole structure. Unlike structured
    D3PM, no reverse target needs to be sampled: the continuous-time ELBO is
    a direct weighted masked-site cross-entropy.

    Returns a per-structure loss of shape ``(batch_size,)``.
    """
    assert hasattr(corruption, "schedule")  # mypy
    assert hasattr(corruption, "mask_index")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy

    device = score_model_output.device
    _validate_loss_configuration(
        score_model_output=score_model_output,
        reduce=reduce,
        charge_of=charge_of,
        mask_idx=mask_idx,
    )

    x0_zero = corruption._to_zero_based(x.long())  # [N_atoms]
    xt_zero = corruption._to_zero_based(noisy_x.long())  # [N_atoms]
    K = score_model_output.shape[-1]
    if bool(((xt_zero < 0) | (xt_zero >= K)).any()):
        raise ValueError(f"noisy species indices must lie in [0, {K - 1}]")

    logits_pad, attention_mask = flat_to_padded(
        score_model_output, batch_idx, batch_size
    )
    x0_pad, _ = flat_to_padded(
        x0_zero, batch_idx, batch_size, fill_value=mask_idx, validate_batch_idx=False
    )
    xt_pad, _ = flat_to_padded(
        xt_zero, batch_idx, batch_size, fill_value=mask_idx, validate_batch_idx=False
    )

    n_sites = attention_mask.sum(dim=1).long()  # [B]
    charge_tensor = torch.tensor(charge_of, dtype=torch.long, device=device)
    q_max = compute_q_max(charge_of, int(n_sites.max().item()))
    _validate_neutral_targets(x0_pad, attention_mask, charge_tensor)

    # Same pinning convention as structured D3PM's denominator. Skips SUBS's
    # own softmax normalisation: the constrained marginal ratio is invariant
    # to a positive per-site rescaling, so pinning the raw logits is equivalent.
    pinned_logits = pin_committed_candidates(logits_pad, xt_pad, attention_mask, mask_idx)

    log_marginals, log_z = neutral_marginals_diff(
        pinned_logits, charge_tensor, q_max, n_sites
    )  # [B, N, K], [B]
    _require_finite_partition(log_z, t=t)

    log_p_true = log_marginals.gather(-1, x0_pad.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # [B, N]
    is_masked = xt_pad == mask_idx
    real_masked = is_masked & attention_mask

    lambda_t = mdlm_loss_weight(corruption.schedule, t)  # [B]; all atoms in a crystal share t
    per_site_loss = torch.where(real_masked, -log_p_true, torch.zeros_like(log_p_true))
    loss = lambda_t * per_site_loss.sum(dim=1)  # [B]

    if reduce == "mean":
        loss = loss / n_sites.clamp(min=1).to(loss.dtype)

    return loss


def make_neutral_mdlm_loss():
    """Factory returning a ``neutral_mdlm_loss`` FieldLoss with charges bound.

    Builds ``charge_of``/``mask_idx`` once from the species vocabulary. Use as
    ``lightning_module.diffusion_module.loss_fn.atomic_numbers_loss_partial``
    (see :class:`mattergen.common.loss.MaterialsLoss`), which supplies
    ``reduce`` itself at call time. Unlike ``make_neutral_d3pm_loss``, there
    are no ablation knobs (``vb_weight``/``ce_weight``/``mc_samples``): the
    continuous-time MDLM ELBO is a single direct term, not a weighted sum of
    a reconstruction and a reverse KL requiring a Monte Carlo estimate.
    """
    charge_of, mask_idx = build_charge_of()
    return partial(neutral_mdlm_loss, charge_of=charge_of, mask_idx=mask_idx)
