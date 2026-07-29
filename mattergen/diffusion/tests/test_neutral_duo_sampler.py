"""test_neutral_duo_sampler.py

Tests for ``mattergen.diffusion.duo.neutral_duo_sampler``:

    * ``NeutralDuoSampler`` -- charge-neutral joint samples, loud RuntimeError
      on infeasibility, vocabulary validation.
    * ``NeutralDuoAncestralSamplingPredictor`` -- instantiates, produces
      finite same-shape output, and the latent it conditions on
      (``NeutralDuoSampler``'s draw) is neutral. Every site remains revisable
      (no visible/masked distinction, unlike D3PM/MDLM).
    * Hydra config instantiation smoke test.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.duo_corruption import DuoCorruption
from mattergen.diffusion.duo.neutral_duo_sampler import (
    NeutralDuoAncestralSamplingPredictor,
    NeutralDuoSampler,
)

CHARGE_OF = [-1, 0, 1]
K = len(CHARGE_OF)


def _corruption() -> DuoCorruption:
    return DuoCorruption(schedule=LogLinearSchedule(), num_classes=K, offset=1)


def _total_charge(tokens: torch.Tensor, charge_of: list[int]) -> int:
    return sum(charge_of[t.item()] for t in tokens)


def _make_flat(batch_sizes: list[int], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_total = sum(batch_sizes)
    logits = torch.randn(n_total, K)
    current_state = torch.randint(0, K, (n_total,))
    batch_idx = torch.cat(
        [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
    )
    return logits, current_state, batch_idx


class TestNeutralDuoSampler:
    def test_charge_neutral_output(self):
        sampler = NeutralDuoSampler(charge_of=CHARGE_OF)
        batch_sizes = [4, 4, 4]
        logits, current_state, batch_idx = _make_flat(batch_sizes, seed=5)
        alpha_u = torch.full((len(batch_sizes),), 0.5)

        for trial in range(20):
            samples = sampler(logits, current_state, alpha_u, batch_idx)
            for b in range(len(batch_sizes)):
                total = _total_charge(samples[batch_idx == b], CHARGE_OF)
                assert total == 0, f"trial {trial}, crystal {b}: charge sum {total}"

    def test_infeasible_raises_loudly(self):
        charge_of = [1, 1, 1]  # all-positive -> no neutral assignment
        sampler = NeutralDuoSampler(charge_of=charge_of)
        logits, current_state, batch_idx = _make_flat([3, 2], seed=9)
        alpha_u = torch.full((2,), 0.5)

        with pytest.raises(RuntimeError, match="no charge-neutral assignment"):
            sampler(logits, current_state, alpha_u, batch_idx)

    def test_hard_excluded_candidate_does_not_rescue_infeasibility(self):
        """A candidate excluded via true -inf (as mask_disallowed_species now
        produces) must not be treated as reachable. Species 1 here is the
        only route to neutrality (species 0 + species 1 = 0); with species 0
        and species 2 alone, no two-site sum reaches zero, so this must
        raise, not silently pick species 1."""
        charge_of = [1, -1, 3]
        sampler = NeutralDuoSampler(charge_of=charge_of)

        logits = torch.tensor([[0.0, float("-inf"), 0.0], [0.0, float("-inf"), 0.0]])
        current_state = torch.zeros(2, dtype=torch.long)
        batch_idx = torch.zeros(2, dtype=torch.long)
        alpha_u = torch.full((1,), 0.5)

        with pytest.raises(RuntimeError, match="no charge-neutral assignment"):
            sampler(logits, current_state, alpha_u, batch_idx)

    def test_logits_must_match_charge_vocabulary(self):
        sampler = NeutralDuoSampler(charge_of=CHARGE_OF)
        with pytest.raises(ValueError, match="logits have 2 classes"):
            sampler(
                torch.randn(2, 2),
                torch.zeros(2, dtype=torch.long),
                torch.ones(1),
                torch.zeros(2, dtype=torch.long),
            )

    def test_default_vocabulary_drops_mask_entry(self):
        from mattergen.constraints.charges import build_charge_of

        full_charge_of, mask_idx = build_charge_of()
        sampler = NeutralDuoSampler()
        assert len(sampler.charge_of) == mask_idx
        assert sampler.charge_of == full_charge_of[:mask_idx]


class TestNeutralDuoAncestralSamplingPredictor:
    def _build(self, charge_of):
        corruption = _corruption()
        predictor = NeutralDuoAncestralSamplingPredictor(
            corruption=corruption, score_fn=(lambda *a, **k: None)
        )
        predictor.neutral_sampler = NeutralDuoSampler(charge_of=charge_of)
        return predictor, corruption

    def test_update_given_score_shapes_finite(self):
        predictor, corruption = self._build(CHARGE_OF)
        batch_sizes = [4, 3]
        n_total = sum(batch_sizes)
        torch.manual_seed(0)
        score = torch.randn(n_total, K)
        x = torch.randint(0, K, (n_total,)) + 1  # 1-based
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )
        t = torch.full((2,), 0.5)

        x_sample, x_expected = predictor.update_given_score(
            x=x, t=t, dt=torch.tensor(-0.05), batch_idx=batch_idx, score=score, batch=None
        )
        assert x_sample.shape == (n_total,)
        assert x_expected.shape == (n_total,)
        assert torch.isfinite(x_sample.float()).all()
        assert torch.isfinite(x_expected.float()).all()

    def test_conditioning_latent_is_neutral(self):
        """The joint sample the predictor conditions its posterior resample on
        must itself be charge-neutral."""
        predictor, corruption = self._build(CHARGE_OF)
        batch_sizes = [4, 4]
        n_total = sum(batch_sizes)
        torch.manual_seed(1)
        score = torch.randn(n_total, K)
        xt_zero = torch.randint(0, K, (n_total,))
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )
        alpha_u = corruption.schedule.alpha(torch.full((len(batch_sizes),), 0.4))

        sampled_s0 = predictor.neutral_sampler(score, xt_zero, alpha_u, batch_idx)
        for b in range(len(batch_sizes)):
            total = _total_charge(sampled_s0[batch_idx == b], CHARGE_OF)
            assert total == 0, f"crystal {b} latent not neutral: {total}"

    def test_final_step_returns_the_sampled_neutral_assignment(self):
        """At r=0 the USDM posterior is deterministic: q(s_0'|s_t,s_0hat) is a
        delta at s_0hat, so the predictor's final-step output must equal the
        sampled neutral assignment exactly."""
        predictor, corruption = self._build(CHARGE_OF)
        n_total = 4
        torch.manual_seed(2)
        score = torch.randn(n_total, K)
        x = torch.randint(0, K, (n_total,)) + 1
        batch_idx = torch.zeros(n_total, dtype=torch.long)
        t = torch.tensor([0.05])
        dt = torch.tensor(-0.05)  # r = t + dt = 0.0 exactly

        torch.manual_seed(100)
        x0_zero = corruption._to_zero_based(x.long())
        expected_s0 = predictor.neutral_sampler(
            score, x0_zero, corruption.schedule.alpha(t), batch_idx
        )
        torch.manual_seed(100)
        x_sample, _ = predictor.update_given_score(
            x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
        )
        x_sample_zero = corruption._to_zero_based(x_sample.long())
        assert torch.equal(x_sample_zero, expected_s0)


def test_sampling_config_instantiates():
    config_path = Path(__file__).parents[3] / "sampling_conf/duo_constrained.yaml"
    config = OmegaConf.load(config_path)
    predictor_partial = hydra.utils.instantiate(
        config.sampler_partial.predictor_partials.atomic_numbers
    )
    assert predictor_partial.func is NeutralDuoAncestralSamplingPredictor
