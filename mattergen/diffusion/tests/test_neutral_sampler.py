"""test_neutral_sampler.py

Tests for ``mattergen.diffusion.sampling.neutral_sampler.NeutralSampler``:
one-hot output, charge-neutral joint samples, committed sites preserved,
loud RuntimeError on infeasibility, MASK never sampled.

Takes flat ``[N_atoms, K]`` logits + a 1-based ``x_t`` + ``batch_idx``
(MatterGen's ChemGraph convention).
"""

from __future__ import annotations

import pytest
import torch

from mattergen.diffusion.sampling.neutral_sampler import NeutralSampler


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


class TestNeutralSampler:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"charge_of": [-1, 0, 1, 0]},
            {"mask_idx": 3},
        ],
    )
    def test_partial_vocabulary_configuration_raises(self, kwargs):
        with pytest.raises(ValueError, match="both be supplied or both be omitted"):
            NeutralSampler(**kwargs)

    def test_mask_must_be_final_zero_charge_class(self):
        with pytest.raises(ValueError, match="final class"):
            NeutralSampler(charge_of=[-1, 0, 1, 0], mask_idx=1)
        with pytest.raises(ValueError, match="charge zero"):
            NeutralSampler(charge_of=[-1, 0, 1, 2], mask_idx=3)

    def test_logits_must_match_charge_vocabulary(self):
        sampler = NeutralSampler(charge_of=[-1, 0, 1, 0], mask_idx=3)
        with pytest.raises(ValueError, match="logits have 3 classes"):
            sampler(
                torch.randn(2, 3),
                torch.full((2,), 4, dtype=torch.long),
                t=torch.tensor(1),
                batch_idx=torch.zeros(2, dtype=torch.long),
            )

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

    def test_true_neginf_excluded_candidate_does_not_rescue_infeasibility(self):
        """A candidate excluded via true -inf must not be treated as reachable
        by the DP. Species 1 here is the only route to neutrality (species 0
        + species 1 = 0); with species 0 and species 2 alone, no two-site sum
        reaches zero, so this must raise, not silently pick species 1.
        """
        charge_of = [1, -1, 3, 0]
        mask_idx = 3
        K = len(charge_of)
        sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)

        logits = torch.tensor(
            [[0.0, float("-inf"), 0.0, 0.0], [0.0, float("-inf"), 0.0, 0.0]]
        )
        x_t = torch.full((2,), mask_idx + 1, dtype=torch.long)
        batch_idx = torch.zeros(2, dtype=torch.long)

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
