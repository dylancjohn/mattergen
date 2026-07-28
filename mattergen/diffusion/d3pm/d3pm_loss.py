# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Literal, Optional

import torch

from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.sde_lib import maybe_expand
from mattergen.diffusion.d3pm.d3pm import compute_kl_reverse_process
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.training.field_loss import aggregate_per_sample


def d3pm_loss(
    *,
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    d3pm_hybrid_lambda: float = 0.0,
    logits_projection_fn: Optional[Callable] = None,
    **_,
) -> torch.Tensor:
    assert hasattr(corruption, "N")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy
    assert hasattr(corruption, "d3pm")  # mypy
    t = maybe_expand(to_discrete_time(t, N=corruption.N, T=corruption.T), batch_idx)

    x0_zero = corruption._to_zero_based(x.long())
    xt_zero = corruption._to_zero_based(noisy_x.long())

    if logits_projection_fn is not None:
        # Project logits through the differentiable SPL layer so that both KL
        # and CE terms see the charge-neutral distribution p_SPL(x_0 | x_t).
        # Gradients flow back through the DP to the raw score_model_output.
        denoiser_logits = logits_projection_fn(score_model_output, xt_zero, batch_idx, batch_size)
    else:
        denoiser_logits = score_model_output

    metrics_dict = compute_kl_reverse_process(
        x0_zero,
        t,
        diffusion=corruption.d3pm,
        log_space=True,
        denoise_fn=lambda targets, timestep: denoiser_logits,
        hybrid_lambda=d3pm_hybrid_lambda,
        x_t_plus_1=xt_zero,
    )
    loss = metrics_dict.pop("loss")
    loss_per_structure = aggregate_per_sample(
        loss, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
    )
    return loss_per_structure
