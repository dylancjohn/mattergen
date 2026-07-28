"""Tests for PredictorCorrector's per-corruption step-count handling
(mattergen.diffusion.sampling.pc_sampler)."""

from __future__ import annotations

import pytest
import torch

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.d3pm.d3pm_predictors_correctors import D3PMAncestralSamplingPredictor
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.mdlm.mdlm_predictors_correctors import MDLMAncestralSamplingPredictor
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector


class _DummyModel(torch.nn.Module):
    def forward(self, x, t):
        return x


class _DummyLoss:
    model_targets: dict = {}


def _diffusion_module(multi_corruption: MultiCorruption) -> DiffusionModule:
    return DiffusionModule(model=_DummyModel(), corruption=multi_corruption, loss_fn=_DummyLoss())


def _predictor_corrector(diffusion_module: DiffusionModule, predictor_cls, N: int) -> PredictorCorrector:
    return PredictorCorrector(
        diffusion_module=diffusion_module,
        predictor_partials={
            "atomic_numbers": lambda corruption, score_fn: predictor_cls(
                corruption=corruption, score_fn=score_fn
            )
        },
        device=torch.device("cpu"),
        n_steps_corrector=0,
        N=N,
    )


def test_mdlm_sampling_n_is_decoupled_from_any_training_time_value():
    mdlm = MDLMCorruption(schedule=LogLinearSchedule(), num_classes=5, offset=1)
    multi_corruption = MultiCorruption(discrete_corruptions={"atomic_numbers": mdlm})
    dm = _diffusion_module(multi_corruption)

    for n in (17, 200, 1000):
        _predictor_corrector(dm, MDLMAncestralSamplingPredictor, N=n)  # must not raise


def test_d3pm_sampling_n_must_still_match_training_time_num_steps():
    schedule = create_discrete_diffusion_schedule(kind="standard", num_steps=37)
    d3pm = D3PMCorruption(d3pm=MaskDiffusion(dim=5, schedule=schedule), offset=1)
    multi_corruption = MultiCorruption(discrete_corruptions={"atomic_numbers": d3pm})
    dm = _diffusion_module(multi_corruption)

    _predictor_corrector(dm, D3PMAncestralSamplingPredictor, N=37)  # matches, fine

    with pytest.raises(AssertionError):
        _predictor_corrector(dm, D3PMAncestralSamplingPredictor, N=38)  # mismatched
