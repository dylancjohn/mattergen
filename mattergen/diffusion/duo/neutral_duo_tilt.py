"""neutral_duo_tilt.py

Forward-likelihood-tilted local weights for constrained Duo.

Duo's candidates are never pinned (every category remains an eligible clean
prediction at every site, unlike the absorbing families' visible/masked
split), so this does not build on
:mod:`mattergen.diffusion.corruption.candidate_pinning`. Parameterised
generically by ``(u, current_state)`` rather than hard-coded to ``(t, s_t)``,
so the same helper covers both the training-time tilt (used by
:mod:`neutral_duo_loss`) and the sampling-time tilt (used by
:mod:`neutral_duo_sampler`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_tilted_local_weights(
    logits: torch.Tensor,
    current_state: torch.LongTensor,
    alpha_u: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """``log w_i(a) = logits_i(a) + log g_u(current_state_i | a)``, where
    ``g_u(y | a) = alpha_u * 1[y=a] + (1 - alpha_u) / K`` is the uniform-state
    forward likelihood.

    Does not exclude padding positions; callers apply that separately.

    Args:
        logits: Per-site clean-category log-scores, shape ``[B, N, K]``.
        current_state: Category observed at time ``u`` to tilt towards, shape
            ``[B, N]``, 0-based. Padding positions may hold any valid index.
        alpha_u: Survival probability at time ``u``, shape ``[B]``.
        num_classes: ``K``, the size of the uniform-state category space.

    Returns:
        Log-weights, shape ``[B, N, K]``.
    """
    K = num_classes
    alpha_u_ = alpha_u.view(-1, 1, 1)  # [B, 1, 1]
    current_one_hot = F.one_hot(current_state.clamp(min=0), K).bool()  # [B, N, K]
    g = torch.where(
        current_one_hot,
        alpha_u_ + (1.0 - alpha_u_) / K,
        (1.0 - alpha_u_) / K * torch.ones_like(logits),
    )
    return logits + g.log()
