"""Forward-likelihood-tilted local weights for constrained Duo.

Duo never pins candidates: every category stays an eligible clean prediction
at every site, unlike the visible/masked split of D3PM and MDLM, so this does
not build on :mod:`mattergen.diffusion.corruption.candidate_pinning`. The
helper takes a generic ``(u, current_state)`` rather than ``(t, s_t)`` so it
serves both the structured loss (:mod:`neutral_duo_loss`) and structured
sampling (:mod:`neutral_duo_sampler`).
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

    ``logits`` ``[B, N, K]`` are clean-category log-scores, ``current_state``
    ``[B, N]`` is the 0-based category observed at time ``u`` and ``alpha_u``
    ``[B]`` is the survival probability at ``u``. Returns ``[B, N, K]``
    log-weights. Padding is not excluded; callers mask it separately, and
    padding positions may hold any valid index.
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
