"""test_neutral_d3pm_sampler.py

Tests for ``NeutralD3PMAncestralSamplingPredictor``: instantiates on a small
MaskDiffusion, ``update_given_score`` returns finite same-shape samples, the
constraint it applies yields a neutral latent x_0, and its empirical reverse
samples match the exact structured mixture.

``NeutralSampler`` itself (the underlying joint sampler this predictor uses)
is tested separately in ``test_neutral_sampler.py``.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.d3pm.neutral_d3pm_sampler import NeutralD3PMAncestralSamplingPredictor
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.sampling.neutral_sampler import NeutralSampler


def _total_charge(tokens: torch.Tensor, charge_of: list[int]) -> int:
    return sum(charge_of[t.item()] for t in tokens)


def _small_corruption(dim: int) -> D3PMCorruption:
    schedule = create_discrete_diffusion_schedule(
        kind="linear", beta_min=1e-3, beta_max=0.1, num_steps=10
    )
    d3pm_obj = MaskDiffusion(dim=dim, schedule=schedule)
    return D3PMCorruption(d3pm=d3pm_obj, offset=1)


class TestNeutralPredictor:
    def _build(self, charge_of, mask_idx):
        corruption = _small_corruption(dim=len(charge_of))
        predictor = NeutralD3PMAncestralSamplingPredictor(
            corruption=corruption, score_fn=(lambda *a, **k: None), predict_x0=True
        )
        # Inject a small synthetic charge vocab in place of the real species vocab.
        predictor.neutral_sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)
        return predictor, corruption

    def test_update_given_score_shapes_finite(self):
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        predictor, corruption = self._build(charge_of, mask_idx)

        batch_sizes = [4, 3]
        n_total = sum(batch_sizes)
        torch.manual_seed(0)
        score = torch.randn(n_total, len(charge_of))
        x = torch.full((n_total,), mask_idx + 1, dtype=torch.long)  # 1-based all-MASK
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )
        t = torch.tensor([0.5, 0.5])

        x_sample, x_expected = predictor.update_given_score(
            x=x, t=t, dt=torch.tensor(0.1), batch_idx=batch_idx, score=score, batch=None
        )
        assert x_sample.shape == (n_total,)
        assert x_expected.shape == (n_total,)
        assert torch.isfinite(x_sample.float()).all()

    def test_applied_constraint_yields_neutral_latent(self):
        """The neutral point-mass the predictor conditions on is a neutral x_0."""
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        predictor, _ = self._build(charge_of, mask_idx)

        batch_sizes = [4, 4]
        n_total = sum(batch_sizes)
        torch.manual_seed(1)
        score = torch.randn(n_total, len(charge_of))
        x = torch.full((n_total,), mask_idx + 1, dtype=torch.long)
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )

        point_mass = predictor.neutral_sampler(score, x, torch.tensor(3), batch_idx)
        x0 = (point_mass == 0.0).float().argmax(dim=-1)  # 0-based latent clean state
        for b in range(len(batch_sizes)):
            total = _total_charge(x0[batch_idx == b], charge_of)
            assert total == 0, f"Latent x_0 for crystal {b} not neutral: {total}"

    def test_reverse_samples_match_exact_structured_mixture(self):
        """The complete predictor samples the intended latent-mixture kernel."""
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        predictor, corruption = self._build(charge_of, mask_idx)
        base_logits = torch.tensor([[0.8, -0.4, 0.1, -2.0], [-0.3, 0.2, 0.9, -1.0]])

        neutral_latents = [
            assignment
            for assignment in itertools.product(range(mask_idx), repeat=2)
            if sum(charge_of[token] for token in assignment) == 0
        ]
        latent_scores = torch.tensor(
            [
                sum(base_logits[i, token].item() for i, token in enumerate(assignment))
                for assignment in neutral_latents
            ]
        )
        latent_probs = torch.softmax(latent_scores, dim=0)

        continuous_t = torch.tensor([0.5])
        discrete_t = int(
            to_discrete_time(continuous_t, N=corruption.N, T=corruption.T)[0].item()
        )
        current = torch.full((2,), mask_idx, dtype=torch.long)
        output_assignments = list(itertools.product(range(len(charge_of)), repeat=2))
        expected = {assignment: 0.0 for assignment in output_assignments}
        for latent_probability, latent in zip(latent_probs, neutral_latents):
            posterior, _, _ = corruption.d3pm.sample_and_compute_posterior_q(
                x_0=torch.tensor(latent, dtype=torch.long),
                t=torch.full((2,), discrete_t, dtype=torch.long),
                samples=current,
                return_logits=False,
                return_transition_probs=True,
            )
            for output in output_assignments:
                conditional = posterior[0, output[0]] * posterior[1, output[1]]
                expected[output] += float(latent_probability * conditional)

        batch_size = 10_000
        score = base_logits.repeat(batch_size, 1)
        x = torch.full((2 * batch_size,), mask_idx + 1, dtype=torch.long)
        batch_idx = torch.arange(batch_size, dtype=torch.long).repeat_interleave(2)
        t = torch.full((batch_size,), 0.5)

        torch.manual_seed(2026)
        samples, _ = predictor.update_given_score(
            x=x,
            t=t,
            dt=torch.tensor(0.1),
            batch_idx=batch_idx,
            score=score,
            batch=None,
        )
        samples = (samples - 1).reshape(batch_size, 2)

        for output, probability in expected.items():
            if probability < 1e-4:
                continue
            empirical = ((samples[:, 0] == output[0]) & (samples[:, 1] == output[1])).float().mean()
            assert empirical.item() == pytest.approx(probability, abs=0.025)
