"""test_neutral_mdlm_sampler.py

Tests for ``mattergen.diffusion.mdlm.neutral_mdlm_sampler``:

    * ``NeutralMDLMAncestralSamplingPredictor`` — instantiates, produces
      finite same-shape output, leaves visible sites untouched, reveals every
      masked site neutrally at the final step, and its reveal probability
      matches the schedule's ``u_{r,t}`` away from the final step.
    * Independently sampling each site from its own constrained marginal can
      produce a non-neutral joint outright, while the predictor's actual
      joint sampling never does. This is why ``NeutralSampler``'s joint
      backward sampling is used instead.

``NeutralSampler`` (the underlying joint sampler, reused unmodified) is
already exhaustively tested in ``test_neutral_sampler.py``; this file only
covers the MDLM-specific reveal-scheduling logic layered on top of it.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from neutral_layer.generation.dp import compute_q_max, neutral_marginals
from omegaconf import OmegaConf

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.candidate_pinning import pin_committed_candidates
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.sampling.neutral_sampler import NeutralSampler
from mattergen.diffusion.mdlm.neutral_mdlm_sampler import NeutralMDLMAncestralSamplingPredictor
from mattergen.diffusion.mdlm.schedule import reveal_prob


def _corruption(num_classes: int) -> MDLMCorruption:
    return MDLMCorruption(schedule=LogLinearSchedule(), num_classes=num_classes, offset=1)


def _build_predictor(charge_of, mask_idx):
    corruption = _corruption(num_classes=len(charge_of))
    predictor = NeutralMDLMAncestralSamplingPredictor(
        corruption=corruption, score_fn=(lambda *a, **k: None)
    )
    predictor.neutral_sampler = NeutralSampler(charge_of=charge_of, mask_idx=mask_idx)
    return predictor, corruption


def _total_charge(tokens: torch.Tensor, charge_of: list[int]) -> int:
    return sum(charge_of[t.item()] for t in tokens)


def test_update_given_score_shapes_finite():
    charge_of = [-1, 0, 1, 0]
    mask_idx = 3
    predictor, _ = _build_predictor(charge_of, mask_idx)

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
        x=x, t=t, dt=torch.tensor(-0.05), batch_idx=batch_idx, score=score, batch=None
    )
    assert x_sample.shape == (n_total,)
    assert x_expected.shape == (n_total,)
    assert torch.isfinite(x_sample.float()).all()
    assert torch.equal(x_sample, x_expected)  # a single joint sample has no separate mean


def test_visible_sites_copied_through_unchanged():
    charge_of = [-1, 0, 1, 0]
    mask_idx = 3
    predictor, _ = _build_predictor(charge_of, mask_idx)

    torch.manual_seed(3)
    score = torch.randn(4, len(charge_of))
    x = torch.tensor([mask_idx + 1, 1, mask_idx + 1, 2])  # atoms 1,3 already visible
    batch_idx = torch.tensor([0, 0, 1, 1])
    t = torch.full((2,), 0.5)

    x_sample, _ = predictor.update_given_score(
        x=x, t=t, dt=torch.tensor(-0.05), batch_idx=batch_idx, score=score, batch=None
    )
    assert x_sample[1].item() == 1
    assert x_sample[3].item() == 2


def test_final_step_reveals_all_masked_sites_neutrally():
    """s = t + dt <= 0 forces every masked site to reveal; the revealed
    values must jointly satisfy charge neutrality (drawn from NeutralSampler,
    not independently per site)."""
    charge_of = [-1, 0, 1, 0]
    mask_idx = 3
    predictor, _ = _build_predictor(charge_of, mask_idx)

    batch_sizes = [4, 4, 4]
    n_total = sum(batch_sizes)
    torch.manual_seed(11)
    score = torch.randn(n_total, len(charge_of))
    x = torch.full((n_total,), mask_idx + 1, dtype=torch.long)
    batch_idx = torch.cat(
        [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(batch_sizes)]
    )
    t = torch.full((3,), 0.02)

    x_sample, _ = predictor.update_given_score(
        x=x, t=t, dt=torch.tensor(-0.05), batch_idx=batch_idx, score=score, batch=None
    )
    x0_zero = x_sample - 1
    assert (x0_zero != mask_idx).all(), "final step must leave no MASK tokens"
    for b in range(len(batch_sizes)):
        total = _total_charge(x0_zero[batch_idx == b], charge_of)
        assert total == 0, f"crystal {b} not neutral: {total}"


def test_reveal_probability_matches_schedule():
    """Away from the final step, each masked site reveals independently with
    probability u_{r,t}; check the empirical reveal rate matches it."""
    charge_of = [-1, 0, 1, 0]
    mask_idx = 3
    predictor, corruption = _build_predictor(charge_of, mask_idx)

    n_crystals = 3000
    torch.manual_seed(21)
    score = torch.randn(n_crystals, len(charge_of)).repeat_interleave(1, dim=0)
    x = torch.full((n_crystals,), mask_idx + 1, dtype=torch.long)
    batch_idx = torch.arange(n_crystals, dtype=torch.long)
    t = torch.full((n_crystals,), 0.5)
    dt = torch.tensor(-0.1)

    expected_u = reveal_prob(corruption.schedule, t[0], (t[0] + dt).clamp(min=0.0)).item()

    x_sample, _ = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    empirical_reveal_rate = (x_sample != mask_idx + 1).float().mean().item()
    assert empirical_reveal_rate == pytest.approx(expected_u, abs=0.03)


def test_independent_marginal_sampling_would_be_wrong_but_predictor_is_always_neutral():
    """Sampling each currently-masked site independently from its
    own constrained marginal can produce a non-neutral joint outright, since
    a one-site marginal does not encode which *combination* of choices is
    jointly valid. With three symmetric non-mask candidates A(-1), B(0),
    C(+1) and two masked sites, the three neutral joint assignments (A,C),
    (C,A), (B,B) are equally likely by symmetry, so every site's constrained
    marginal is uniform (1/3, 1/3, 1/3) -- but independently drawing from
    that marginal at each site lands on a non-neutral pair (e.g. (A,A)) about
    two-thirds of the time. The predictor's actual finite-step sampler must
    never do this: it always draws one joint neutral assignment first.
    """
    charge_of = [-1, 0, 1, 0]  # A, B, C, MASK
    mask_idx = 3
    K = len(charge_of)

    base_logits = torch.zeros(1, 2, K)
    base_logits[:, :, mask_idx] = float("-inf")
    xt_pad = torch.full((1, 2), mask_idx, dtype=torch.long)
    attention_mask = torch.ones(1, 2, dtype=torch.bool)
    charge_tensor = torch.tensor(charge_of, dtype=torch.long)
    q_max = compute_q_max(charge_of, 2)

    pinned = pin_committed_candidates(base_logits, xt_pad, attention_mask, mask_idx)
    log_marg, log_z = neutral_marginals(pinned, charge_tensor, q_max, torch.tensor([2]))
    assert log_z.isfinite().all()
    probs = log_marg[0].exp()  # [2, K]
    assert torch.allclose(probs[:, :3], torch.full((2, 3), 1 / 3), atol=1e-6)

    torch.manual_seed(0)
    n_trials = 4000
    independent_site0 = torch.multinomial(probs[0], n_trials, replacement=True)
    independent_site1 = torch.multinomial(probs[1], n_trials, replacement=True)
    independent_neutral = torch.tensor(
        [
            charge_of[a] + charge_of[b] == 0
            for a, b in zip(independent_site0.tolist(), independent_site1.tolist())
        ]
    )
    # True rate is 1/3; a generous upper bound well below "always neutral"
    # is enough to demonstrate the failure without being a flaky exact check.
    assert independent_neutral.float().mean().item() < 0.6

    predictor, _ = _build_predictor(charge_of, mask_idx)
    n_crystals = 500
    score = base_logits[0].repeat(n_crystals, 1)
    x = torch.full((2 * n_crystals,), mask_idx + 1, dtype=torch.long)
    batch_idx = torch.arange(n_crystals, dtype=torch.long).repeat_interleave(2)
    t = torch.full((n_crystals,), 0.02)

    x_sample, _ = predictor.update_given_score(
        x=x, t=t, dt=torch.tensor(-0.05), batch_idx=batch_idx, score=score, batch=None
    )
    x0_zero = (x_sample - 1).reshape(n_crystals, 2)
    predictor_neutral = (
        charge_tensor[x0_zero[:, 0]] + charge_tensor[x0_zero[:, 1]] == 0
    )
    assert predictor_neutral.all(), "predictor produced a non-neutral joint sample"


def test_sampling_config_instantiates():
    config_path = (
        Path(__file__).parents[3] / "sampling_conf/mdlm_constrained.yaml"
    )
    config = OmegaConf.load(config_path)
    predictor_partial = hydra.utils.instantiate(
        config.sampler_partial.predictor_partials.atomic_numbers
    )
    assert predictor_partial.func is NeutralMDLMAncestralSamplingPredictor
