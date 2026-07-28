# Adapted from https://github.com/kuleshov-group/mdlm, which is released
# under the Apache License, Version 2.0.

"""Unconstrained MDLM training objective:

    L_MDLM = lambda_MDLM(t) * sum_{i: x_t,i = MASK} -log p_theta(x_0,i | x_t, t)

Assumes uniform time sampling (pi(t) = 1), so no importance-sampling
reweighting term is applied.
"""

from typing import Literal

import torch

from mattergen.diffusion.corruption.corruption import Corruption, maybe_expand
from mattergen.diffusion.mdlm.schedule import mdlm_loss_weight
from mattergen.diffusion.mdlm.subs import true_clean_state_log_prob
from mattergen.diffusion.training.field_loss import aggregate_per_sample


def mdlm_loss(
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
    assert hasattr(corruption, "mask_index")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy

    x0_zero = corruption._to_zero_based(x.long())
    xt_zero = corruption._to_zero_based(noisy_x.long())

    t_per_atom = maybe_expand(t, batch_idx)
    lambda_t = mdlm_loss_weight(corruption.schedule, t_per_atom)

    log_p_true = true_clean_state_log_prob(
        score_model_output, xt_zero, x0_zero, corruption.mask_index
    )

    is_masked = xt_zero == corruption.mask_index
    per_atom_loss = torch.where(
        is_masked, -log_p_true * lambda_t, torch.zeros_like(log_p_true)
    )

    return aggregate_per_sample(
        per_atom_loss, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
    )
