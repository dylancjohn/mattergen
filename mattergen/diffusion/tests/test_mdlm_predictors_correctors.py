from __future__ import annotations

import torch

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.mdlm.mdlm_predictors_correctors import MDLMAncestralSamplingPredictor

K = 5
MASK_IDX = K - 1


def _corruption() -> MDLMCorruption:
    return MDLMCorruption(schedule=LogLinearSchedule(eps=1e-3), num_classes=K, offset=1)


def _predictor(corruption: MDLMCorruption) -> MDLMAncestralSamplingPredictor:
    return MDLMAncestralSamplingPredictor(corruption=corruption, score_fn=None)


def test_is_compatible_only_with_mdlm_corruption():
    mdlm_corruption = _corruption()
    assert MDLMAncestralSamplingPredictor.is_compatible(mdlm_corruption)

    d3pm_corruption = D3PMCorruption(
        d3pm=MaskDiffusion(
            dim=K, schedule=create_discrete_diffusion_schedule(kind="standard", num_steps=10)
        ),
        offset=1,
    )
    assert not MDLMAncestralSamplingPredictor.is_compatible(d3pm_corruption)


def test_visible_sites_are_invariant_under_reverse_step():
    torch.manual_seed(0)
    corruption = _corruption()
    predictor = _predictor(corruption)

    x = torch.tensor([1, 2, 3])  # all visible (1-based, non-mask)
    batch_idx = torch.tensor([0, 0, 0])
    t = torch.full((1,), 0.5)
    dt = torch.tensor(-0.01)
    score = torch.randn(3, K)

    x_next, x_expected = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    assert torch.equal(x_next, x)
    assert torch.equal(x_expected, x)


def test_final_interval_forces_full_reveal():
    torch.manual_seed(1)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 50
    x = torch.full((n,), MASK_IDX + 1, dtype=torch.long)  # all masked (1-based)
    batch_idx = torch.arange(n)
    t = torch.full((n,), 0.01)
    dt = torch.tensor(-0.02)  # s = t + dt < 0: final interval
    score = torch.randn(n, K)

    x_next, x_expected = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    x_next_zero = corruption._to_zero_based(x_next)
    x_expected_zero = corruption._to_zero_based(x_expected)
    assert (x_next_zero != MASK_IDX).all()
    assert (x_expected_zero != MASK_IDX).all()


def test_revealed_sites_never_resample_to_mask():
    torch.manual_seed(2)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 2000
    x = torch.full((n,), MASK_IDX + 1, dtype=torch.long)
    batch_idx = torch.arange(n)
    # A longer interval so plenty of sites actually get revealed this step.
    t = torch.full((n,), 0.9)
    dt = torch.tensor(-0.5)
    score = torch.randn(n, K)

    x_next, _ = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    # "Revealed" = changed from the original (all-MASK) state -- an independent
    # signal from x_next's own value, so this isn't circular.
    changed = x_next != x
    assert changed.any(), "test setup should reveal at least some sites"
    x_next_zero = corruption._to_zero_based(x_next)
    assert (x_next_zero[changed] != MASK_IDX).all()  # SUBS excludes MASK as a clean category


def test_freeze_once_revealed_across_two_steps():
    torch.manual_seed(3)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 500
    x = torch.full((n,), MASK_IDX + 1, dtype=torch.long)
    batch_idx = torch.arange(n)
    score = torch.randn(n, K)

    t1 = torch.full((n,), 0.6)
    dt1 = torch.tensor(-0.2)
    x_after_1, _ = predictor.update_given_score(
        x=x, t=t1, dt=dt1, batch_idx=batch_idx, score=score, batch=None
    )
    revealed_after_1 = corruption._to_zero_based(x_after_1) != MASK_IDX

    t2 = torch.full((n,), 0.4)
    dt2 = torch.tensor(-0.2)
    score2 = torch.randn(n, K)
    x_after_2, _ = predictor.update_given_score(
        x=x_after_1, t=t2, dt=dt2, batch_idx=batch_idx, score=score2, batch=None
    )

    # Sites revealed after step 1 must be unchanged after step 2.
    assert torch.equal(x_after_2[revealed_after_1], x_after_1[revealed_after_1])


def test_reveal_frequency_matches_u_rt_statistically():
    torch.manual_seed(4)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 200_000
    x = torch.full((n,), MASK_IDX + 1, dtype=torch.long)
    batch_idx = torch.arange(n)
    t = torch.full((n,), 0.6)
    dt = torch.tensor(-0.1)
    score = torch.randn(n, K)

    x_next, _ = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    empirical_reveal_freq = (corruption._to_zero_based(x_next) != MASK_IDX).float().mean().item()

    s = t[0] + dt
    expected_u = (
        (corruption.schedule.alpha(s) - corruption.schedule.alpha(t[0]))
        / (1.0 - corruption.schedule.alpha(t[0]))
    ).item()
    assert abs(empirical_reveal_freq - expected_u) < 5e-3
