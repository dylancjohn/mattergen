"""Tests for mattergen.diffusion.continuous_time.schedule.

Covers, for every Schedule subclass:
    * alpha(0) ~= 1 and alpha(1) is small (bounded away from the exact
      endpoints rather than reaching them; see schedule.py's module docstring
      for which endpoint each schedule's eps/sigma_min/sigma_max controls).
    * alpha is strictly decreasing on [0, 1].
    * dalpha_dt matches a central finite-difference estimate of alpha.
    * alpha_ratio(t, r) == alpha(t) / alpha(r).
    * evaluating at the exact endpoints t=0, t=1 produces finite values
      (no NaN/Inf), including for dalpha_dt.
"""

from __future__ import annotations

import pytest
import torch

from mattergen.diffusion.continuous_time.schedule import (
    CosineSchedule,
    CosineSqrSchedule,
    GeometricSchedule,
    LinearSchedule,
    LogLinearSchedule,
    Schedule,
)

SCHEDULES: list[Schedule] = [
    LogLinearSchedule(),
    CosineSchedule(),
    CosineSqrSchedule(),
    LinearSchedule(),
    GeometricSchedule(),
]


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_alpha_at_t0_is_one(schedule: Schedule):
    # LogLinear/Cosine/CosineSqr clip via an explicit `eps` and hit alpha(0)==1
    # exactly; Linear/Geometric are parameterised through sigma_min instead, so
    # alpha(0) = exp(-sigma_min) is only approximately 1 (matching the MDLM
    # reference implementation's own convention, not a bug in the port).
    t0 = torch.zeros(1, dtype=torch.float64)
    assert torch.allclose(schedule.alpha(t0), torch.ones(1, dtype=torch.float64), atol=2e-3)


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_alpha_at_t1_is_small_and_finite(schedule: Schedule):
    t1 = torch.ones(1, dtype=torch.float64)
    alpha_1 = schedule.alpha(t1)
    assert torch.isfinite(alpha_1).all()
    assert (alpha_1 < 0.05).all()
    assert (alpha_1 >= 0).all()


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_alpha_strictly_decreasing(schedule: Schedule):
    t = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    alpha = schedule.alpha(t)
    assert (alpha[1:] < alpha[:-1]).all()


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_dalpha_dt_matches_finite_difference(schedule: Schedule):
    t = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    h = 1e-6
    numerical = (schedule.alpha(t + h) - schedule.alpha(t - h)) / (2 * h)
    analytical = schedule.dalpha_dt(t)
    assert torch.allclose(analytical, numerical, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_dalpha_dt_is_nonpositive(schedule: Schedule):
    t = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    assert (schedule.dalpha_dt(t) <= 0).all()


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_alpha_ratio_matches_direct_division(schedule: Schedule):
    r = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
    t = torch.tensor([0.4, 0.6, 0.9], dtype=torch.float64)
    ratio = schedule.alpha_ratio(t, r)
    expected = schedule.alpha(t) / schedule.alpha(r)
    assert torch.allclose(ratio, expected)


@pytest.mark.parametrize("schedule", SCHEDULES, ids=lambda s: type(s).__name__)
def test_endpoints_are_finite_including_derivative(schedule: Schedule):
    t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert torch.isfinite(schedule.alpha(t)).all()
    assert torch.isfinite(schedule.dalpha_dt(t)).all()


def test_loglinear_default_eps():
    schedule = LogLinearSchedule()
    assert schedule.eps == 1e-3


def test_loglinear_is_affine():
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    expected = 1.0 - (1.0 - schedule.eps) * t
    assert torch.allclose(schedule.alpha(t), expected)


def test_loglinear_dalpha_dt_is_constant():
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    dalpha = schedule.dalpha_dt(t)
    assert torch.allclose(dalpha, dalpha[0].expand_as(dalpha))
    assert torch.allclose(dalpha[0], torch.tensor(-(1.0 - schedule.eps), dtype=torch.float64))
