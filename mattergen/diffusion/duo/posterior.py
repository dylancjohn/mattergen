# Ported near-verbatim from Duo's `DUO_BASE._posterior_from_x0`
# (https://github.com/s-sahoo/duo), which is released under the Apache
# License, Version 2.0. Adapted here to MatterGen's flat [N_atoms, K]
# per-atom tensor convention rather than a padded [batch, length, K] tensor.

"""USDM analytical posterior and mean parameterisation for Duo.

The one-site uniform-state forward posterior ``q(s_r = b | s_t = c, s_0 = a)``
is, in vector form:

    pi_{r|t}(z, x) = [K*alpha_t*(z . x) + (alpha_{t|r} - alpha_t)*z
                       + (alpha_r - alpha_t)*x + (1 - alpha_{t|r})*(1 - alpha_r)/K]
                      / [K*alpha_t*<z, x> + 1 - alpha_t]

where ``z = e_{s_t}`` (one-hot) and ``x`` is the clean-state distribution
(one-hot for the true clean state, or -- Duo's USDM mean parameterisation --
the denoiser's predicted simplex ``p_theta(x_0 | s_t, t)`` substituted
directly in place of the one-hot). This substitution is nonlinear: it must
not be replaced by the generic D3PM mixture over clean categories.
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
    """Return ``q(s_r | s_t, x0_probs)`` under the USDM mean parameterisation.

    Args:
        x0_probs: predicted (or true one-hot) clean-state distribution, flat
            ``[N_atoms, K]``, 0-based category axis. Simplex-valued rows.
        xt: current noisy category per atom, flat ``[N_atoms]``, 0-based.
        alpha_r, alpha_t: survival probabilities at the earlier (``r``) and
            later (``t``) times, flat ``[N_atoms]``, with ``alpha_r >= alpha_t``.
        num_classes: ``K``, the size of the uniform-state category space.

    Returns:
        Flat ``[N_atoms, K]`` posterior distribution over ``s_r``, summing to
        1 along the last axis (up to floating-point error).
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
