# Adapted from MDLM's `_subs_parameterization` (mdlm/diffusion.py):
# https://github.com/kuleshov-group/mdlm, released under the Apache License,
# Version 2.0. Adapted here to MatterGen's flat [N_atoms, K] per-atom tensor
# convention rather than a padded [batch, length, K] sequence tensor.

"""SUBS (substitution) reverse parameterisation for MDLM.

Two substitutions are applied to the raw denoiser logits over the ``K``
categories (``K - 1`` clean species plus MASK at the last index): the MASK
category is given zero probability as a clean-state output, and an
already-visible site is copied through exactly as a one-hot distribution at
its observed category, bypassing the network for that site.
"""

import torch

NEG_INFINITY = -1e6


def subs_log_probs(
    raw_logits: torch.Tensor,
    xt_zero: torch.Tensor,
    mask_index: int,
) -> torch.Tensor:
    """Return ``log p_theta(x_0 | x_t)`` under the SUBS parameterisation.

    Args:
        raw_logits: denoiser output, flat ``[N_atoms, K]``, 0-based category axis.
        xt_zero: current noisy category per atom, flat ``[N_atoms]``, 0-based.
        mask_index: 0-based index of the MASK category (``K - 1``).

    Returns:
        Flat ``[N_atoms, K]`` log-probabilities: an exact one-hot delta at
        ``xt_zero`` for visible (non-mask) atoms, and the network's
        mask-suppressed, renormalised distribution for masked atoms.
    """
    logits = raw_logits.clone()
    logits[:, mask_index] = logits[:, mask_index] + NEG_INFINITY
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    is_visible = xt_zero != mask_index
    if is_visible.any():
        visible_one_hot = torch.full_like(log_probs[is_visible], NEG_INFINITY)
        visible_one_hot.scatter_(-1, xt_zero[is_visible].unsqueeze(-1), 0.0)
        log_probs = log_probs.clone()
        log_probs[is_visible] = visible_one_hot

    return log_probs


def true_clean_state_log_prob(
    raw_logits: torch.Tensor,
    xt_zero: torch.Tensor,
    x0_zero: torch.Tensor,
    mask_index: int,
) -> torch.Tensor:
    """``log p_theta(x_0 = x0_zero | x_t)`` per atom, under SUBS."""
    log_probs = subs_log_probs(raw_logits, xt_zero, mask_index)
    return log_probs.gather(-1, x0_zero.unsqueeze(-1)).squeeze(-1)
