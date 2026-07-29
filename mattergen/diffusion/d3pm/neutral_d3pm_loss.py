"""neutral_d3pm_loss.py

Structured charge-neutral training objective for MatterGen's absorbing D3PM.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from neutral_layer.generation.dp import compute_q_max, flat_to_padded, neutral_log_z
from torch.distributions import Categorical

from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.corruption.candidate_pinning import pin_committed_candidates
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.discrete_time import to_discrete_time


def _validate_mc_samples(mc_samples: int) -> None:
    if isinstance(mc_samples, bool) or not isinstance(mc_samples, int) or mc_samples < 1:
        raise ValueError(f"mc_samples must be a positive integer, got {mc_samples!r}")


def _validate_loss_configuration(
    *,
    score_model_output: torch.Tensor,
    reduce: str,
    mc_samples: int,
    charge_of: list[int],
    mask_idx: int,
) -> None:
    """Validate the absorbing-D3PM vocabulary and loss options."""
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
    _validate_mc_samples(mc_samples)
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
            "absorbing D3PM requires MASK to be the final class; "
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
            "neutral_d3pm_loss requires charge-neutral clean targets; "
            f"invalid structure charge sums: {totals}"
        )


def _require_finite_partition(
    log_z: torch.Tensor,
    *,
    name: str,
    t_disc: torch.LongTensor,
) -> None:
    """Raise with batch context instead of propagating a non-finite loss."""
    finite = torch.isfinite(log_z)
    if bool(finite.all()):
        return
    bad = (~finite).nonzero(as_tuple=True)[0].tolist()
    times = {b: int(t_disc[b].item()) for b in bad}
    values = {b: float(log_z[b].detach().item()) for b in bad}
    raise RuntimeError(
        f"neutral_d3pm_loss has non-finite {name} partition for structures {bad}; "
        f"discrete times={times}, log_z={values}. Check the species vocabulary, "
        "pinned assignments, and model logits."
    )


def _numerator_logits(
    den_logits: torch.Tensor,
    xt_pad: torch.LongTensor,
    s_prev_pad: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    mask_idx: int,
) -> torch.Tensor:
    """Build the numerator's effective logits for the sampled reverse target.

    Absorbing fast-path: at each newly-revealed site (``x_t == MASK`` and the
    sampled ``x_{t-1} != MASK``, whose value equals the true ``x_0``), keep only
    the revealed species — masking every other species to ``-inf`` while
    retaining the model's *raw* logit at the revealed species (so the gradient
    ``∂ log Z_num / ∂ ℓ_i(v) = ρ_i(v)`` flows).  Still-masked sites keep
    ``den_logits`` unchanged; committed sites are already pinned there.
    """
    K = den_logits.shape[-1]
    revealed = (
        (xt_pad == mask_idx) & (s_prev_pad != mask_idx) & attention_mask
    )  # [B, N]

    revealed_one_hot = F.one_hot(s_prev_pad.clamp(min=0), K).bool()  # [B, N, K]
    keep = torch.where(
        revealed.unsqueeze(-1), revealed_one_hot, torch.ones_like(revealed_one_hot)
    )
    return torch.where(keep, den_logits, torch.full_like(den_logits, float("-inf")))


def neutral_d3pm_loss(
    *,
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    vb_weight: float,
    ce_weight: float,
    mc_samples: int,
    charge_of: list[int],
    mask_idx: int,
    **_,
) -> torch.Tensor:
    """Structured charge-neutral absorbing-D3PM loss for atomic numbers.

    The reverse term omits absorbing-posterior factors that are constant with
    respect to the compatible latent clean assignment.  Its value is therefore
    equal to the structured reverse NLL only up to a theta-independent additive
    constant, while its gradient is exact.  MatterGen's hybrid clean-state CE is
    applied at every sampled time, including the reconstruction time, matching
    :func:`mattergen.diffusion.d3pm.d3pm.compute_kl_reverse_process`.

    Returns a per-structure loss of shape ``(batch_size,)``.
    """
    assert hasattr(corruption, "N")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy
    assert hasattr(corruption, "d3pm")  # mypy

    device = score_model_output.device
    _validate_loss_configuration(
        score_model_output=score_model_output,
        reduce=reduce,
        mc_samples=mc_samples,
        charge_of=charge_of,
        mask_idx=mask_idx,
    )

    want_ce = ce_weight != 0.0
    want_vb = vb_weight != 0.0
    if not (want_ce or want_vb):
        return score_model_output.new_zeros(batch_size)

    # Per-structure discrete time (all atoms in a crystal share t).
    t_disc = to_discrete_time(t, N=corruption.N, T=corruption.T).long()  # [B]

    x0_zero = corruption._to_zero_based(x.long())  # [N_atoms]
    xt_zero = corruption._to_zero_based(noisy_x.long())  # [N_atoms]
    K = score_model_output.shape[-1]
    if bool(((xt_zero < 0) | (xt_zero >= K)).any()):
        raise ValueError(f"noisy species indices must lie in [0, {K - 1}]")

    # Flat → padded.  Logits keep gradients through the index assignment.
    logits_pad, attention_mask = flat_to_padded(
        score_model_output, batch_idx, batch_size
    )
    x0_pad, _ = flat_to_padded(
        x0_zero,
        batch_idx,
        batch_size,
        fill_value=mask_idx,
        validate_batch_idx=False,
    )
    xt_pad, _ = flat_to_padded(
        xt_zero,
        batch_idx,
        batch_size,
        fill_value=mask_idx,
        validate_batch_idx=False,
    )

    n_sites = attention_mask.sum(dim=1).long()  # [B]
    charge_tensor = torch.tensor(charge_of, dtype=torch.long, device=device)
    q_max = compute_q_max(charge_of, int(n_sites.max().item()))
    _validate_neutral_targets(x0_pad, attention_mask, charge_tensor)

    den_logits = pin_committed_candidates(logits_pad, xt_pad, attention_mask, mask_idx)

    # Shared denominator log-partition (differentiable).
    log_z_den = neutral_log_z(den_logits, charge_tensor, q_max, n_sites)  # [B]
    _require_finite_partition(log_z_den, name="denominator", t_disc=t_disc)

    # ── CE term: constrained clean-state NLL = log Z_den − Σ_i ℓ_i(x_0*,i) ──
    # den_logits at the true label: 0 for committed sites, raw logit for masked.
    true_logit = den_logits.gather(-1, x0_pad.clamp(min=0).unsqueeze(-1)).squeeze(
        -1
    )  # [B, N]
    bad_true_score = attention_mask & ~torch.isfinite(true_logit)
    if bool(bad_true_score.any()):
        bad_structures = bad_true_score.any(dim=1).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(
            "neutral_d3pm_loss found a non-finite score for the clean target in "
            f"structures {bad_structures}"
        )
    true_logit = torch.where(attention_mask, true_logit, torch.zeros_like(true_logit))
    l_ce = log_z_den - true_logit.sum(dim=1)  # [B]

    loss = score_model_output.new_zeros(batch_size)
    if want_ce:
        # Deliberately includes t_disc == 0, matching base MatterGen hybrid CE.
        loss = loss + ce_weight * l_ce

    # ── VB reverse term: mean over MC samples of (log Z_den − log Z_num) ──
    if want_vb:
        t_atom = t_disc[batch_idx]  # [N_atoms]
        is_t0_atom = t_atom == 0  # reconstruction reveals every site

        l_vb_acc = score_model_output.new_zeros(batch_size)
        for _s in range(mc_samples):
            # Sample the reverse target exactly as compute_kl_reverse_process does.
            q_prev, _, _ = corruption.d3pm.sample_and_compute_posterior_q(
                x_0=x0_zero,
                t=t_atom,
                make_one_hot=True,
                samples=xt_zero,
                return_logits=False,
                return_transition_probs=True,
            )  # [N_atoms, K]
            s_prev = Categorical(probs=q_prev).sample()  # [N_atoms], 0-based
            # t==0: force a full reveal to x_0 (so vb collapses to ce).
            s_prev = torch.where(is_t0_atom, x0_zero, s_prev)

            s_prev_pad, _ = flat_to_padded(
                s_prev,
                batch_idx,
                batch_size,
                fill_value=mask_idx,
                validate_batch_idx=False,
            )
            num_logits = _numerator_logits(
                den_logits, xt_pad, s_prev_pad, attention_mask, mask_idx
            )
            log_z_num = neutral_log_z(num_logits, charge_tensor, q_max, n_sites)  # [B]
            _require_finite_partition(
                log_z_num, name=f"numerator (MC sample {_s})", t_disc=t_disc
            )
            l_vb_acc = l_vb_acc + (log_z_den - log_z_num)

        l_vb = l_vb_acc / mc_samples
        loss = loss + vb_weight * l_vb

    if reduce == "mean":
        loss = loss / n_sites.clamp(min=1).to(loss.dtype)

    return loss


def make_neutral_d3pm_loss(
    vb_weight: float,
    ce_weight: float,
    mc_samples: int = 1,
):
    """Factory returning a ``neutral_d3pm_loss`` FieldLoss with charges bound.

    Builds ``charge_of``/``mask_idx`` once from the species vocabulary. Use as
    ``lightning_module.diffusion_module.loss_fn.atomic_numbers_loss_partial``
    (see :class:`mattergen.common.loss.MaterialsLoss`), which supplies
    ``reduce`` itself at call time.
    """
    _validate_mc_samples(mc_samples)
    charge_of, mask_idx = build_charge_of()
    return partial(
        neutral_d3pm_loss,
        vb_weight=vb_weight,
        ce_weight=ce_weight,
        mc_samples=mc_samples,
        charge_of=charge_of,
        mask_idx=mask_idx,
    )
