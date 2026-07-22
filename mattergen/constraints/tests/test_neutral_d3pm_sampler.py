"""test_neutral_d3pm_sampler.py

Tests for the inference-time charge-neutrality glue in
``mattergen.constraints.neutral_d3pm_sampler``:

    * ``NeutralSampler`` — one-hot output, charge-neutral joint samples,
      committed sites preserved, loud RuntimeError on infeasibility, MASK never
      sampled.
    * ``NeutralD3PMAncestralSamplingPredictor`` — instantiates on a small
      MaskDiffusion, ``update_given_score`` returns finite same-shape samples,
      and the constraint it applies yields a neutral latent x_0.

``NeutralSampler`` takes flat ``[N_atoms, K]`` logits + a 1-based ``x_t`` +
``batch_idx`` (MatterGen's ChemGraph convention).
"""

from __future__ import annotations

import pytest
import torch

from mattergen.constraints.neutral_d3pm_sampler import (
    NeutralD3PMAncestralSamplingPredictor,
    NeutralSampler,
)
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flat(
    batch_sizes: list[int],
    K: int,
    mask_idx: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat [N_atoms, K] logits, all-MASK 1-based x_t, and batch_idx."""
    torch.manual_seed(seed)
    n_total = sum(batch_sizes)
    logits = torch.randn(n_total, K)
    x_t = torch.full((n_total,), mask_idx + 1, dtype=torch.long)  # 1-based MASK
    batch_idx = torch.cat(
        [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
    )
    return logits, x_t, batch_idx


def _total_charge(tokens: torch.Tensor, charge_of: list[int]) -> int:
    return sum(charge_of[t.item()] for t in tokens)


# ---------------------------------------------------------------------------
# NeutralSampler
# ---------------------------------------------------------------------------


class TestNeutralSampler:
    def test_one_hot_output_at_real_atoms(self):
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        logits, x_t, batch_idx = _make_flat([4, 3, 2], K, mask_idx)
        result = sampler(logits, x_t, t=10, batch_idx=batch_idx)

        for atom in range(result.shape[0]):
            n_zeros = (result[atom] == 0.0).sum().item()
            n_neginf = (result[atom] == float("-inf")).sum().item()
            assert n_zeros == 1, f"Atom {atom}: expected 1 zero, got {n_zeros}"
            assert n_neginf == K - 1

    def test_charge_neutral_output(self):
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        batch_sizes = [4, 4, 4, 4]
        torch.manual_seed(5)
        logits = torch.randn(sum(batch_sizes), K)
        logits[:, mask_idx] = float("-inf")
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )
        x_t = torch.full((sum(batch_sizes),), mask_idx + 1, dtype=torch.long)

        for trial in range(20):
            result = sampler(logits, x_t, t=5, batch_idx=batch_idx)
            tokens = (result == 0.0).float().argmax(dim=-1)
            for b in range(len(batch_sizes)):
                total = _total_charge(tokens[batch_idx == b], charge_of)
                assert total == 0, f"Trial {trial}, crystal {b}: charge sum = {total}"

    def test_committed_sites_preserved(self):
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        logits, x_t, batch_idx = _make_flat([4, 4], K, mask_idx, seed=1)
        x_t[0] = 1  # crystal 0, atom 0 committed to species 0 (1-based → 1)
        x_t[4] = 1  # crystal 1, atom 0 committed to species 0

        result = sampler(logits, x_t, t=2, batch_idx=batch_idx)

        for atom_idx in [0, 4]:
            assert result[atom_idx, 0].item() == pytest.approx(0.0)
            for s in range(K):
                if s != 0:
                    assert result[atom_idx, s].item() == float("-inf")

    def test_infeasible_raises_loudly(self):
        """No charge-neutral assignment must raise RuntimeError, not degrade silently."""
        charge_of = [1, 1, 1, 0]  # all-positive real species → infeasible
        mask_idx = 3
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        torch.manual_seed(9)
        batch_sizes = [3, 2]
        logits = torch.randn(sum(batch_sizes), K)
        logits[:, mask_idx] = float("-inf")
        x_t = torch.full((sum(batch_sizes),), mask_idx + 1, dtype=torch.long)
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )

        with pytest.raises(RuntimeError, match="no charge-neutral assignment"):
            sampler(logits, x_t, t=1, batch_idx=batch_idx)

    def test_mask_token_never_sampled(self):
        charge_of = [-1, 0, 1, 0]
        mask_idx = 3
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        logits, x_t, batch_idx = _make_flat([4, 3, 4], K, mask_idx)
        for _ in range(20):
            result = sampler(logits, x_t, t=5, batch_idx=batch_idx)
            tokens = (result == 0.0).float().argmax(dim=-1)
            assert (tokens != mask_idx).all(), "MASK token was sampled"

    def test_multi_crystal_batch_all_neutral(self):
        charge_of = [-2, -1, 0, 1, 2, 0]
        mask_idx = 5
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        batch_sizes = [3, 5, 4, 2]
        torch.manual_seed(42)
        logits = torch.randn(sum(batch_sizes), K)
        logits[:, mask_idx] = float("-inf")
        x_t = torch.full((sum(batch_sizes),), mask_idx + 1, dtype=torch.long)
        batch_idx = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
        )

        for trial in range(15):
            result = sampler(logits, x_t, t=3, batch_idx=batch_idx)
            tokens = (result == 0.0).float().argmax(dim=-1)
            for b in range(len(batch_sizes)):
                total = _total_charge(tokens[batch_idx == b], charge_of)
                assert total == 0, f"Trial {trial}, crystal {b}: charge sum = {total}"


# ---------------------------------------------------------------------------
# NeutralD3PMAncestralSamplingPredictor
# ---------------------------------------------------------------------------


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
