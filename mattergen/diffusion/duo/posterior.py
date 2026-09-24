# Ported near-verbatim from Duo's `DUO_BASE._posterior_from_x0`
# (https://github.com/s-sahoo/duo), which is released under the Apache
# License, Version 2.0. Adapted to MatterGen's flat [N_atoms, K] per-atom
# tensors rather than padded [batch, length, K] tensors.

"""Uniform-state (USDM) posterior with Duo's mean parameterisation.

The one-site forward posterior ``q(s_r = b | s_t = c, s_0 = a)``, ``r < t``,
is, in vector form:

    pi_{r|t}(z, x) = [K*alpha_t*(z * x) + (alpha_{t|r} - alpha_t)*z
                       + (alpha_r - alpha_t)*x + (1 - alpha_{t|r})*(1 - alpha_r)/K]
                      / [K*alpha_t*<z, x> + 1 - alpha_t]

where ``z = e_{s_t}`` is one-hot, ``*`` is elementwise and ``x`` is the
clean-state distribution: one-hot for the true clean state, or, under Duo's
mean parameterisation, the denoiser's predicted simplex
``p_theta(x_0 | s_t, t)`` substituted directly for the one-hot. The
substitution is nonlinear, so it is not equivalent to the D3PM-style mixture
of one-hot posteriors weighted by ``p_theta``.
"""

import torch
import torch.nn.functional as F


def usdm_posterior(
    x0_probs: torch.Tensor,
    xt: torch.Tensor,
    alpha_r: torch.Tensor,
    alpha_t: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return ``q(s_r | s_t, x0_probs)`` as a flat ``[N_atoms, K]`` distribution.

    ``x0_probs`` ``[N_atoms, K]`` has simplex rows (predicted or true one-hot);
    ``xt`` ``[N_atoms]`` holds 0-based noisy categories; ``alpha_r`` and
    ``alpha_t`` ``[N_atoms]`` are survival probabilities at the earlier and
    later times, with ``alpha_r >= alpha_t``. Rows sum to 1 up to rounding.
    """
    K = num_classes
    xt_one_hot = F.one_hot(xt, K).to(x0_probs.dtype)  # [N, K]

    alpha_r_ = alpha_r.unsqueeze(-1)
    alpha_t_ = alpha_t.unsqueeze(-1)
    alpha_t_given_r = alpha_t_ / alpha_r_
    d_alpha = alpha_r_ - alpha_t_

    numerator = (
        alpha_t_ * K * x0_probs * xt_one_hot
        + (alpha_t_given_r - alpha_t_) * xt_one_hot
        + d_alpha * x0_probs
        + (1 - alpha_t_given_r) * (1 - alpha_r_) / K
    )
    x0_at_xt = torch.gather(x0_probs, -1, xt.unsqueeze(-1))  # [N, 1]
    denominator = alpha_t_ * K * x0_at_xt + (1 - alpha_t_)
    return numerator / denominator
