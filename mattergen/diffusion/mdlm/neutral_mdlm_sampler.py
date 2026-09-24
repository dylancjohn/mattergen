"""Structured (charge-neutral) sampling for MDLM atom-type diffusion.

Joint neutral clean-state sampling uses the same pinned absorbing-mask
convention as structured D3PM (visible sites are singleton candidates, masked
sites keep all non-mask species), so
:class:`mattergen.diffusion.sampling.neutral_sampler.NeutralSampler` is
reused as is.
"""

from __future__ import annotations

from typing import Optional

import torch

from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.corruption.sde_lib import ScoreFunction
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.mdlm.mdlm_predictors_correctors import MDLMAncestralSamplingPredictor
from mattergen.diffusion.mdlm.schedule import reveal_prob
from mattergen.diffusion.sampling.neutral_sampler import NeutralSampler
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class NeutralMDLMAncestralSamplingPredictor(MDLMAncestralSamplingPredictor):
    """MDLM ancestral predictor that enforces charge neutrality.

    Each reverse step (1) draws one complete neutral clean assignment from
    the pinned charge-neutral distribution via :class:`NeutralSampler`, then
    (2) reveals each masked site from that assignment independently with
    probability ``u_{r,t}``, as in the base class. Taking revealed values from
    one joint sample keeps the correlations that neutrality imposes between
    sites, which independent per-site sampling would lose.
    """

    def __init__(
        self,
        *,
        corruption: MDLMCorruption,
        score_fn: ScoreFunction,
    ):
        super().__init__(corruption=corruption, score_fn=score_fn)
        self.neutral_sampler = NeutralSampler()

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: Optional[BatchedData],
    ) -> SampleAndMean:
        assert isinstance(self.corruption, MDLMCorruption)
        corruption = self.corruption

        xt_zero = corruption._to_zero_based(x.long())
        t_per_atom = maybe_expand(t, batch_idx)
        # dt is a 0-d scalar shared by the batch, so plain broadcasting is
        # used; maybe_expand needs a batch dimension that dt lacks.
        s_per_atom = t_per_atom + dt

        is_masked = xt_zero == corruption.mask_index
        force_reveal = s_per_atom <= 0
        u = reveal_prob(corruption.schedule, t_per_atom, s_per_atom.clamp(min=0.0))
        reveal = is_masked & (force_reveal | (torch.rand_like(u) < u))

        # One-hot logits at a jointly sampled neutral clean assignment. x is
        # passed 1-based; NeutralSampler handles the offset.
        class_logits = self.neutral_sampler(score, x, t, batch_idx)
        revealed_clean = torch.argmax(class_logits, dim=-1)  # 0-based

        # There is no separate posterior mean for a joint sample, so the
        # sample is also reported as the mean.
        x_next = torch.where(reveal, revealed_clean, xt_zero)
        x_expected = x_next

        return (
            corruption._to_non_zero_based(x_next),
            corruption._to_non_zero_based(x_expected),
        )
