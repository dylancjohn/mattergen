# Ancestral sampler adapted from https://github.com/s-sahoo/duo, which is
# released under the Apache License, Version 2.0.

from typing import Optional

import torch

from mattergen.diffusion.corruption.corruption import Corruption, maybe_expand
from mattergen.diffusion.corruption.duo_corruption import DuoCorruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.duo.posterior import usdm_posterior
from mattergen.diffusion.sampling.predictors import Predictor
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class DuoAncestralSamplingPredictor(Predictor):
    """Duo ancestral sampler.

    Evaluates the denoiser once on the complete current assignment and
    resamples every site from the finite-interval USDM posterior kernel.
    There is no masked/visible distinction: every ordinary category remains
    eligible for revision at every step.
    """

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return isinstance(corruption, DuoCorruption)

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
        assert isinstance(self.corruption, DuoCorruption)
        corruption = self.corruption

        xt_zero = corruption._to_zero_based(x.long())
        t_per_atom = maybe_expand(t, batch_idx)
        r_per_atom = (t_per_atom + dt).clamp(min=0.0)  # dt <= 0, so r <= t

        x0_probs = torch.softmax(score, dim=-1)
        alpha_t = corruption.schedule.alpha(t_per_atom)
        alpha_r = corruption.schedule.alpha(r_per_atom)

        posterior = usdm_posterior(x0_probs, xt_zero, alpha_r, alpha_t, corruption.num_classes)
        posterior = posterior.clamp(min=0.0)

        sample = torch.distributions.Categorical(probs=posterior).sample()
        expected = torch.argmax(posterior, dim=-1)

        return (
            corruption._to_non_zero_based(sample),
            corruption._to_non_zero_based(expected),
        )
