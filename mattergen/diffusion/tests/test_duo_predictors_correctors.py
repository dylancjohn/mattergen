from __future__ import annotations

import torch

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.duo_corruption import DuoCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.duo.duo_predictors_correctors import DuoAncestralSamplingPredictor

K = 4


def _corruption() -> DuoCorruption:
    return DuoCorruption(schedule=LogLinearSchedule(eps=1e-3), num_classes=K, offset=1)


def _predictor(corruption: DuoCorruption) -> DuoAncestralSamplingPredictor:
    return DuoAncestralSamplingPredictor(corruption=corruption, score_fn=None)


def test_is_compatible_only_with_duo_corruption():
    duo_corruption = _corruption()
    assert DuoAncestralSamplingPredictor.is_compatible(duo_corruption)

    d3pm_corruption = D3PMCorruption(
        d3pm=MaskDiffusion(
            dim=K + 1, schedule=create_discrete_diffusion_schedule(kind="standard", num_steps=10)
        ),
        offset=1,
    )
    assert not DuoAncestralSamplingPredictor.is_compatible(d3pm_corruption)


def test_every_site_can_change_even_if_not_masked():
    """Unlike MDLM, Duo has no visible/masked distinction: any site, whatever
    its current (ordinary, non-mask) category, can change at every step."""
    torch.manual_seed(0)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 5000
    x = torch.randint(1, K + 1, (n,))  # arbitrary ordinary categories, 1-based
    batch_idx = torch.arange(n)
    t = torch.full((n,), 0.9)
    dt = torch.tensor(-0.3)
    score = torch.randn(n, K)

    x_next, _ = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    assert (x_next != x).any(), "some sites should change over a long, noisy interval"


def test_output_categories_are_in_range():
    torch.manual_seed(1)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 500
    x = torch.randint(1, K + 1, (n,))
    batch_idx = torch.arange(n)
    t = torch.full((n,), 0.5)
    dt = torch.tensor(-0.1)
    score = torch.randn(n, K)

    x_next, x_expected = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    x_next_zero = corruption._to_zero_based(x_next)
    x_expected_zero = corruption._to_zero_based(x_expected)
    assert (x_next_zero >= 0).all() and (x_next_zero < K).all()
    assert (x_expected_zero >= 0).all() and (x_expected_zero < K).all()


def test_final_interval_concentrates_on_confident_prediction():
    """At r=0 (the terminal reverse step), a strongly confident denoiser
    prediction should dominate the sampled category most of the time."""
    torch.manual_seed(2)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 2000
    confident_category = 2  # 0-based
    x = torch.randint(1, K + 1, (n,))
    batch_idx = torch.arange(n)
    t = torch.full((n,), 0.02)
    dt = torch.tensor(-0.02)  # r = t + dt = 0.0, terminal reveal
    score = torch.full((n, K), -10.0)
    score[:, confident_category] = 10.0

    x_next, x_expected = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    x_expected_zero = corruption._to_zero_based(x_expected)
    assert (x_expected_zero == confident_category).all()

    x_next_zero = corruption._to_zero_based(x_next)
    frac_confident = (x_next_zero == confident_category).float().mean().item()
    assert frac_confident > 0.9


def test_expected_state_is_argmax_of_posterior():
    torch.manual_seed(3)
    corruption = _corruption()
    predictor = _predictor(corruption)

    n = 100
    x = torch.randint(1, K + 1, (n,))
    batch_idx = torch.arange(n)
    t = torch.full((n,), 0.6)
    dt = torch.tensor(-0.2)
    score = torch.randn(n, K)

    _, x_expected = predictor.update_given_score(
        x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=None
    )
    # Recompute the posterior directly and check consistency with the
    # predictor's reported "expected" (mean) state.
    from mattergen.diffusion.corruption.corruption import maybe_expand
    from mattergen.diffusion.duo.posterior import usdm_posterior

    xt_zero = corruption._to_zero_based(x)
    t_per_atom = maybe_expand(t, batch_idx)
    r_per_atom = (t_per_atom + dt).clamp(min=0.0)
    x0_probs = torch.softmax(score, dim=-1)
    alpha_t = corruption.schedule.alpha(t_per_atom)
    alpha_r = corruption.schedule.alpha(r_per_atom)
    posterior = usdm_posterior(x0_probs, xt_zero, alpha_r, alpha_t, K)
    expected_manual = torch.argmax(posterior, dim=-1)

    assert torch.equal(corruption._to_zero_based(x_expected), expected_manual)
