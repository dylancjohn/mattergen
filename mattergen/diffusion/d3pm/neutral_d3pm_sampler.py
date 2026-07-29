"""neutral_d3pm_sampler.py

Inference-time charge-neutrality constraint for MatterGen's absorbing D3PM.
"""

from __future__ import annotations

from typing import Optional

import torch

from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.sde_lib import ScoreFunction
from mattergen.diffusion.d3pm.d3pm_predictors_correctors import (
    D3PMAncestralSamplingPredictor,
)
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.sampling.neutral_sampler import NeutralSampler
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class NeutralD3PMAncestralSamplingPredictor(D3PMAncestralSamplingPredictor):
    """Ancestral D3PM predictor that enforces charge neutrality.

    Applies :class:`NeutralSampler` to the model logits (yielding a jointly
    charge-neutral point-mass ``x_0``), then runs the same ``predict_x0`` reverse
    step as the base predictor, so the reverse kernel is the exact structured
    kernel ``sum_{x_0 in V} q(x_{t-1} | x_t, x_0) p̃_θ(x_0 | x_t)``.

    Config-injectable exactly like the stock predictor (needs only
    ``predict_x0: True`` and ``_partial_: true``); the ``NeutralSampler`` is built
    internally from :func:`build_charge_of`.
    """

    def __init__(
        self,
        *,
        corruption: D3PMCorruption,
        score_fn: ScoreFunction,
        predict_x0: bool = True,
    ):
        super().__init__(
            corruption=corruption, score_fn=score_fn, predict_x0=predict_x0
        )
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
        # t is continuous, needs to be integer.
        t = to_discrete_time(t=t, N=self.N, T=self.corruption.T)

        assert isinstance(self.corruption, D3PMCorruption)

        # Apply the charge-neutrality constraint: replace the model logits with
        # one-hot logits of a jointly-sampled neutral x_0.  x is 1-based here;
        # NeutralSampler handles the offset internally.
        class_logits = self.neutral_sampler(score, x, t, batch_idx)

        # sample from categorical distribution
        x_sample = self.corruption._to_non_zero_based(
            torch.distributions.Categorical(logits=class_logits).sample()
        )

        # convert logit output to normalized probabilities
        class_probs = torch.softmax(class_logits, dim=-1)

        # get expected atom type from categorical distribution
        class_expected = self.corruption._to_non_zero_based(
            torch.argmax(class_probs, dim=-1)
        )

        if self.predict_x0:
            # the (constrained) model predicts p(x_0|x_t); evaluate p(x_{t-1}|x_t)
            # by Eq. 4 in https://arxiv.org/pdf/2107.03006v1.pdf.
            class_logits, _ = self.corruption.d3pm.sample_and_compute_posterior_q(
                x_0=class_probs,
                t=t[batch_idx].to(torch.long),  # requires torch.long or torch.int32
                make_one_hot=False,
                samples=self.corruption._to_zero_based(
                    x
                ),  # d3pm expects 0 offset atom type integers
                return_logits=True,
            )

            x_sample = self.corruption._to_non_zero_based(
                torch.distributions.Categorical(logits=class_logits).sample()
            )

            # get expected atom type
            class_expected = self.corruption._to_non_zero_based(
                torch.argmax(
                    torch.softmax(class_logits.to(class_probs.dtype), dim=-1), dim=-1
                )
            )

        # (sampled states), (expected states)
        return x_sample, class_expected
