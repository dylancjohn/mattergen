# Ported near-verbatim from Duo's `DUO_BASE.nll_per_token`
# (https://github.com/s-sahoo/duo), which is released under the Apache
# License, Version 2.0. Adapted here to MatterGen's flat [N_atoms, K]
# per-atom convention (no [batch, length] axis) and to alpha(t)/alpha'(t)
# supplied by a mattergen.diffusion.continuous_time.schedule.Schedule.

"""Unconstrained Duo training objective: the closed-form, Rao-Blackwellised
``f_Duo`` continuous-time NELBO.

The rate-KL sum ``lambda_Duo(t) * sum_{b != s_t} d_rate(rho*(b) || rho_theta(b))``
is algebraically fused into two closed-form terms (``term1``, ``term2``
below) rather than materialised as an explicit loop over categories ``b``.
"""

from typing import Literal

import torch

from mattergen.diffusion.corruption.corruption import Corruption, maybe_expand
from mattergen.diffusion.training.field_loss import aggregate_per_sample


def _duo_rate_nll(
    log_x_theta: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    alpha_t: torch.Tensor,
    dalpha_t: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Per-atom Rao-Blackwellised rate-KL NELBO term.

    Args:
        log_x_theta: denoiser log-probabilities over clean categories, flat
            ``[N_atoms, K]``.
        xt, x0: current noisy / clean category per atom, flat ``[N_atoms]``,
            0-based.
        alpha_t, dalpha_t: ``alpha(t)``, ``alpha'(t)`` per atom, flat ``[N_atoms]``.
        num_classes: ``K``.
    """
    K = num_classes
    x_reconst = log_x_theta.exp()
    x_bar_theta = K * alpha_t.unsqueeze(-1) * x_reconst + (1 - alpha_t.unsqueeze(-1))
    coeff = dalpha_t / (K * alpha_t)  # == -lambda_Duo(t)

    x_eq_xt = (x0 == xt).to(x_reconst.dtype)
    x_neq_xt = 1.0 - x_eq_xt

    xbar_xt = (1 - alpha_t) + K * alpha_t * x_eq_xt
    xbar_theta_xt = x_bar_theta.gather(-1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = x_bar_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    term1 = K * (1.0 / xbar_xt - 1.0 / xbar_theta_xt)

    const = (1 - alpha_t) / (K * alpha_t + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((K - 1) * const * x_eq_xt - (1.0 / const) * x_neq_xt) * const.log()
    term2_theta = -term2_coefs * (x_bar_theta.log().sum(-1) - K * xbar_theta_xt.log())
    term2_theta = term2_theta - K * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()
    ) * x_neq_xt
    term2 = term2_theta + term2_offset

    return coeff * (term1 - term2)


def duo_loss(
    *,
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    **_,
) -> torch.Tensor:
    assert hasattr(corruption, "schedule")  # mypy
    assert hasattr(corruption, "num_classes")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy

    x0_zero = corruption._to_zero_based(x.long())
    xt_zero = corruption._to_zero_based(noisy_x.long())

    t_per_atom = maybe_expand(t, batch_idx)
    alpha_t = corruption.schedule.alpha(t_per_atom)
    dalpha_t = corruption.schedule.dalpha_dt(t_per_atom)

    log_x_theta = torch.log_softmax(score_model_output, dim=-1)
    per_atom_loss = _duo_rate_nll(
        log_x_theta, xt_zero, x0_zero, alpha_t, dalpha_t, corruption.num_classes
    )

    return aggregate_per_sample(
        per_atom_loss, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
    )
