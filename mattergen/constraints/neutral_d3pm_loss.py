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

from mattergen.common.loss import MaterialsLoss
from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.discrete_time import to_discrete_time


def _pin_denominator_logits(
    logits_pad: torch.Tensor,
    xt_pad: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    mask_idx: int,
) -> torch.Tensor:
    """Build the denominator's pinned logits (autograd-safe).

    Committed sites (``x_t != MASK``) are collapsed to a one-hot at their known
    clean value; the MASK column and padding are set to ``-inf``; masked real
    sites keep their raw logits so gradients flow only there.
    """
    K = logits_pad.shape[-1]
    committed = xt_pad != mask_idx  # [B, N]; padding is MASK-filled → not committed

    # Committed → one-hot at the committed (== clean) species.
    committed_one_hot = torch.full_like(logits_pad, float("-inf"))
    committed_one_hot.scatter_(-1, xt_pad.clamp(min=0).unsqueeze(-1), 0.0)
    pinned = torch.where(committed.unsqueeze(-1), committed_one_hot, logits_pad)

    # Exclude the MASK column (never a valid clean species).
    mask_col = F.one_hot(torch.tensor(mask_idx, device=logits_pad.device), K).bool()
    pinned = torch.where(mask_col, torch.full_like(pinned, float("-inf")), pinned)

    # Zero out padding.
    pinned = torch.where(
        attention_mask.unsqueeze(-1), pinned, torch.full_like(pinned, float("-inf"))
    )
    return pinned


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
    """Structured charge-neutral D3PM loss for the atomic-numbers field.

    Returns a per-structure loss of shape ``(batch_size,)``.
    """
    assert hasattr(corruption, "N")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy
    assert hasattr(corruption, "d3pm")  # mypy

    device = score_model_output.device

    # Per-structure discrete time (all atoms in a crystal share t).
    t_disc = to_discrete_time(t, N=corruption.N, T=corruption.T).long()  # [B]

    x0_zero = corruption._to_zero_based(x.long())  # [N_atoms]
    xt_zero = corruption._to_zero_based(noisy_x.long())  # [N_atoms]

    # Flat → padded.  Logits keep gradients through the index assignment.
    logits_pad, attention_mask = flat_to_padded(
        score_model_output, batch_idx, batch_size
    )
    x0_pad, _ = flat_to_padded(x0_zero, batch_idx, batch_size, fill_value=mask_idx)
    xt_pad, _ = flat_to_padded(xt_zero, batch_idx, batch_size, fill_value=mask_idx)

    n_sites = attention_mask.sum(dim=1).long()  # [B]
    charge_tensor = torch.tensor(charge_of, dtype=torch.long, device=device)
    q_max = compute_q_max(charge_of, int(n_sites.max().item()))

    den_logits = _pin_denominator_logits(logits_pad, xt_pad, attention_mask, mask_idx)

    want_ce = ce_weight != 0.0
    want_vb = vb_weight != 0.0
    if not (want_ce or want_vb):
        return torch.zeros(batch_size, device=device)

    # Shared denominator log-partition (differentiable).
    log_z_den = neutral_log_z(den_logits, charge_tensor, q_max, n_sites)  # [B]

    # ── CE term: constrained clean-state NLL = log Z_den − Σ_i ℓ_i(x_0*,i) ──
    # den_logits at the true label: 0 for committed sites, raw logit for masked.
    true_logit = den_logits.gather(-1, x0_pad.clamp(min=0).unsqueeze(-1)).squeeze(
        -1
    )  # [B, N]
    true_logit = torch.where(attention_mask, true_logit, torch.zeros_like(true_logit))
    l_ce = log_z_den - true_logit.sum(dim=1)  # [B]

    loss = torch.zeros(batch_size, device=device)
    if want_ce:
        loss = loss + ce_weight * l_ce

    # ── VB reverse term: mean over MC samples of (log Z_den − log Z_num) ──
    if want_vb:
        t_atom = t_disc[batch_idx]  # [N_atoms]
        is_t0_atom = t_atom == 0  # reconstruction reveals every site

        l_vb_acc = torch.zeros(batch_size, device=device)
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
                s_prev, batch_idx, batch_size, fill_value=mask_idx
            )
            num_logits = _numerator_logits(
                den_logits, xt_pad, s_prev_pad, attention_mask, mask_idx
            )
            log_z_num = neutral_log_z(num_logits, charge_tensor, q_max, n_sites)  # [B]
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
    reduce: Literal["sum", "mean"] = "mean",
):
    """Factory returning a ``neutral_d3pm_loss`` FieldLoss with charges bound.

    Builds ``charge_of``/``mask_idx`` once from the species vocabulary.  Use as
    the ``atomic_numbers`` field loss (see :class:`NeutralMaterialsLoss`).
    """
    charge_of, mask_idx = build_charge_of()
    return partial(
        neutral_d3pm_loss,
        reduce=reduce,
        vb_weight=vb_weight,
        ce_weight=ce_weight,
        mc_samples=mc_samples,
        charge_of=charge_of,
        mask_idx=mask_idx,
    )


class NeutralMaterialsLoss(MaterialsLoss):
    """``MaterialsLoss`` with the constrained structured atom-type objective.

    Identical to ``MaterialsLoss`` for the continuous (pos/cell) fields; the
    ``atomic_numbers`` field uses :func:`neutral_d3pm_loss` instead of the
    factorised ``d3pm_loss``.  Config-swappable via ``loss_fn._target_``.

    Parameters
    ----------
    vb_weight, ce_weight
        Weights of the structured VB reverse term and the constrained
        clean-state CE term.  A weight of 0 skips that term entirely.
    mc_samples
        Monte-Carlo samples of the reverse target for the VB term.
    **kwargs
        Forwarded to :class:`MaterialsLoss` (must include
        ``include_atomic_numbers=True``, the default).
    """

    def __init__(
        self,
        *,
        vb_weight: float,
        ce_weight: float,
        mc_samples: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert "atomic_numbers" in self.loss_fns, (
            "NeutralMaterialsLoss requires include_atomic_numbers=True"
        )
        self.loss_fns["atomic_numbers"] = make_neutral_d3pm_loss(
            vb_weight=vb_weight,
            ce_weight=ce_weight,
            mc_samples=mc_samples,
            reduce=self.reduce,
        )
