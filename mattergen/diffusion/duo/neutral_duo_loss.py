"""neutral_duo_loss.py

Structured charge-neutral training objective for MatterGen's continuous-time
Duo.

Duo's clean-state prediction is nonlinearly substituted into the uniform-state
posterior, so the constrained loss is not a simple masked-marginal
cross-entropy as for D3PM/MDLM: it replaces the entire per-site prediction by
a forward-likelihood-*tilted* charge-neutral joint distribution, then compares
the true and model reverse-rate ratios site by site.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from neutral_layer.generation.dp import compute_q_max, flat_to_padded, neutral_marginals_diff

from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.duo.neutral_duo_tilt import build_tilted_local_weights
from mattergen.diffusion.duo.schedule import duo_jump_rate


def _validate_loss_configuration(
    *, score_model_output: torch.Tensor, reduce: str, charge_of: list[int], num_classes: int
) -> None:
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
    if score_model_output.ndim != 2:
        raise ValueError(
            "score_model_output must have shape [N_atoms, K], got "
            f"{score_model_output.shape}"
        )
    K = score_model_output.shape[-1]
    if K != num_classes:
        raise ValueError(f"corruption.num_classes={num_classes} does not match logits' K={K}")
    if len(charge_of) != K:
        raise ValueError(
            f"charge_of has length {len(charge_of)}, but logits have {K} classes"
        )


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
            "neutral_duo_loss requires charge-neutral clean targets; "
            f"invalid structure charge sums: {totals}"
        )


def _require_finite_partition(log_z: torch.Tensor, *, t: torch.Tensor) -> None:
    finite = torch.isfinite(log_z)
    if bool(finite.all()):
        return
    bad = (~finite).nonzero(as_tuple=True)[0].tolist()
    times = {b: float(t[b].item()) for b in bad}
    raise RuntimeError(
        f"neutral_duo_loss has non-finite tilted partition for structures {bad}; "
        f"t={times}. Check the species vocabulary and model logits."
    )


def neutral_duo_loss(
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
    **_,
) -> torch.Tensor:
    """Structured charge-neutral Duo loss for atomic numbers.

    Builds the forward-likelihood-tilted local weights ``w_i(a) = ell_i(a) *
    g_t(s_t[i] | a)``, runs one DP pass to get the tilted one-site marginals
    ``omega_i``, forms the model rate ratio ``v_{i,b}`` from those marginals
    in closed form, and compares it to the true two-case rate ratio ``u*_{i,b}``
    via the rate-KL sum. A single forward-backward pass provides every needed
    ``v_{i,b}``; no per-``(i,b)`` DP call is made.

    Returns a per-structure loss of shape ``(batch_size,)``.
    """
    assert hasattr(corruption, "schedule")  # mypy
    assert hasattr(corruption, "num_classes")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy

    device = score_model_output.device
    _validate_loss_configuration(
        score_model_output=score_model_output,
        reduce=reduce,
        charge_of=charge_of,
        num_classes=corruption.num_classes,
    )

    x0_zero = corruption._to_zero_based(x.long())  # [N_atoms]
    xt_zero = corruption._to_zero_based(noisy_x.long())  # [N_atoms]
    K = score_model_output.shape[-1]
    if bool(((xt_zero < 0) | (xt_zero >= K)).any()):
        raise ValueError(f"noisy species indices must lie in [0, {K - 1}]")

    logits_pad, attention_mask = flat_to_padded(score_model_output, batch_idx, batch_size)
    x0_pad, _ = flat_to_padded(
        x0_zero, batch_idx, batch_size, fill_value=0, validate_batch_idx=False
    )
    xt_pad, _ = flat_to_padded(
        xt_zero, batch_idx, batch_size, fill_value=0, validate_batch_idx=False
    )

    n_sites = attention_mask.sum(dim=1).long()  # [B]
    charge_tensor = torch.tensor(charge_of, dtype=torch.long, device=device)
    q_max = compute_q_max(charge_of, int(n_sites.max().item()))
    _validate_neutral_targets(x0_pad, attention_mask, charge_tensor)

    alpha_t = corruption.schedule.alpha(t)  # [B]
    A_t = corruption.num_classes * alpha_t  # [B]
    R_t = 1.0 - alpha_t  # [B]
    lambda_t = duo_jump_rate(corruption.schedule, t, corruption.num_classes)  # [B]

    tilted_logits = build_tilted_local_weights(logits_pad, xt_pad, alpha_t, corruption.num_classes)
    tilted_logits = torch.where(
        attention_mask.unsqueeze(-1), tilted_logits, torch.full_like(tilted_logits, float("-inf"))
    )

    log_marg, log_z = neutral_marginals_diff(
        tilted_logits, charge_tensor, q_max, n_sites
    )  # [B, N, K], [B]
    _require_finite_partition(log_z, t=t)
    omega = log_marg.exp()  # [B, N, K]; -inf (padding) -> 0

    R_over_ApR = (R_t / (A_t + R_t)).view(-1, 1, 1)
    ApR_over_R = ((A_t + R_t) / R_t).view(-1, 1, 1)
    omega_c = omega.gather(-1, xt_pad.unsqueeze(-1))  # [B, N, 1]

    # v_b = omega_b*(A+R)/R + omega_c*R/(A+R) + (1 - omega_b - omega_c): an
    # explicit non-negative mixture, algebraically identical to
    # 1 + (A/R)*omega_b - (A/(A+R))*omega_c but without subtracting two
    # close-in-magnitude terms as t -> 0 (A/R -> inf), which let the
    # cancellation-prone form underflow to a spuriously negative v under
    # float32 rounding (confirmed empirically at K=428).
    term_b = torch.nan_to_num(omega * ApR_over_R, nan=0.0)  # guards 0*inf as t->0
    term_c = omega_c * R_over_ApR
    residual = (1.0 - omega - omega_c).clamp(min=0.0)  # fp rounding guard (exact for b != c)
    v = term_b + term_c + residual  # [B, N, K]; non-negative by construction
    v = v.clamp(min=torch.finfo(v.dtype).eps)  # avoid log(0) if a term is exactly 0

    c_eq_astar = xt_pad == x0_pad  # [B, N]
    is_astar_col = F.one_hot(x0_pad, K).bool()  # [B, N, K]

    u_star = torch.ones_like(v)
    u_star = torch.where(c_eq_astar.unsqueeze(-1), R_over_ApR.expand_as(v), u_star)
    u_star = torch.where(
        (~c_eq_astar).unsqueeze(-1) & is_astar_col, ApR_over_R.expand_as(v), u_star
    )

    d_rate = u_star * (u_star.log() - v.log()) - u_star + v  # [B, N, K]
    not_diag = ~F.one_hot(xt_pad, K).bool()  # exclude b == c
    d_rate = torch.where(not_diag, d_rate, torch.zeros_like(d_rate))
    d_rate = torch.where(attention_mask.unsqueeze(-1), d_rate, torch.zeros_like(d_rate))

    per_site = d_rate.sum(dim=-1)  # [B, N]
    loss = lambda_t * per_site.sum(dim=1)  # [B]

    if reduce == "mean":
        loss = loss / n_sites.clamp(min=1).to(loss.dtype)

    return loss


def make_neutral_duo_loss():
    """Factory returning a ``neutral_duo_loss`` FieldLoss with charges bound.

    Duo has no MASK class, so the trailing MASK entry that
    :func:`build_charge_of` always appends is dropped: Duo's vocabulary is
    exactly the ``num_species`` non-mask categories.
    """
    charge_of, mask_idx = build_charge_of()
    duo_charge_of = charge_of[:mask_idx]
    return partial(neutral_duo_loss, charge_of=duo_charge_of)
