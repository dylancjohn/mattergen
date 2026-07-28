# Ancestral sampler adapted from https://github.com/kuleshov-group/mdlm,
# which is released under the Apache License, Version 2.0.

from typing import Optional

import torch

from mattergen.diffusion.corruption.corruption import Corruption, maybe_expand
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.mdlm.schedule import reveal_prob
from mattergen.diffusion.mdlm.subs import subs_log_probs
from mattergen.diffusion.sampling.predictors import Predictor
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class MDLMAncestralSamplingPredictor(Predictor):
    """MDLM ancestral sampler: freeze-once-revealed reveal steps.

    Visible sites are copied through exactly. Each masked site is revealed
    with probability ``u_{r,t}``; once revealed, a clean category is sampled
    from the SUBS-parameterised denoiser distribution (MASK excluded) and the
    site is never touched again. At the final reverse interval, every
    remaining masked site is forced to reveal deterministically, so the final
    sample contains no MASK tokens.
    """

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return isinstance(corruption, MDLMCorruption)

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
        # dt is a 0-d scalar shared by the whole batch, so ordinary broadcasting
        # against t_per_atom is correct here; maybe_expand assumes a batch
        # dimension, which a 0-d dt does not have.
        s_per_atom = t_per_atom + dt  # dt <= 0, so s <= t

        log_probs = subs_log_probs(score, xt_zero, corruption.mask_index)

        is_masked = xt_zero == corruption.mask_index
        force_reveal = s_per_atom <= 0
        u = reveal_prob(corruption.schedule, t_per_atom, s_per_atom.clamp(min=0.0))
        reveal = is_masked & (force_reveal | (torch.rand_like(u) < u))

        sampled_clean = torch.distributions.Categorical(logits=log_probs).sample()
        expected_clean = torch.argmax(log_probs, dim=-1)

        x_next = torch.where(reveal, sampled_clean, xt_zero)
        x_expected = torch.where(reveal, expected_clean, xt_zero)

        return (
            corruption._to_non_zero_based(x_next),
            corruption._to_non_zero_based(x_expected),
        )
