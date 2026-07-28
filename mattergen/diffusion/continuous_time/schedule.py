# Schedule classes adapted from https://github.com/kuleshov-group/mdlm
# (noise_schedule.py), which is released under the Apache License, Version 2.0.

"""Continuous-time noise schedules shared by the MDLM and Duo atom-type
diffusion families: a strictly decreasing survival coefficient ``alpha(t)``
on ``[0, 1]``, with ``alpha(0) ~= 1`` and ``alpha(1) ~= 0`` (bounded away from
the exact endpoints rather than reaching them), plus its derivative and
finite-interval ratio.

``LogLinearSchedule``/``CosineSchedule``/``CosineSqrSchedule`` reach
``alpha(0) = 1`` exactly and ``alpha(1) = eps > 0``. ``LinearSchedule``/
``GeometricSchedule`` instead approach both endpoints only approximately:
``alpha(0) = exp(-sigma_min)`` and ``alpha(1) = exp(-sigma_max)``. This
matches the reference implementations' own convention rather than reaching
the exact endpoints, at the cost of a correspondingly small bias.
"""

from __future__ import annotations

import abc
import math

import torch


class Schedule(abc.ABC):
    """Strictly decreasing survival-probability schedule ``alpha: [0, 1] -> [0, 1]``."""

    @abc.abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``alpha(t)``."""

    @abc.abstractmethod
    def dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``d(alpha)/dt`` evaluated at ``t``. Always ``<= 0``."""

    def alpha_ratio(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Return ``alpha(t) / alpha(r)`` for ``r <= t`` (finite-interval survival ratio)."""
        return self.alpha(t) / self.alpha(r)


class LogLinearSchedule(Schedule):
    """``alpha(t) = 1 - (1 - eps) * t``. The default schedule for both families."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - self.eps) * t

    def dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, -(1.0 - self.eps))


class CosineSchedule(Schedule):
    """``alpha(t) = eps + (1 - eps) * cos(t * pi / 2)``.

    Validated against MDLM only. Duo's own cosine schedule uses a different
    functional form; do not select this schedule for Duo without checking
    numerical agreement first.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.eps + (1.0 - self.eps) * torch.cos(t * (math.pi / 2))

    def dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        return -(1.0 - self.eps) * (math.pi / 2) * torch.sin(t * (math.pi / 2))


class CosineSqrSchedule(Schedule):
    """``alpha(t) = eps + (1 - eps) * cos(t * pi / 2) ** 2``. Validated against MDLM only."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        c = torch.cos(t * (math.pi / 2))
        return self.eps + (1.0 - self.eps) * c * c

    def dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        half_pi = math.pi / 2
        return -(1.0 - self.eps) * 2 * torch.cos(t * half_pi) * torch.sin(t * half_pi) * half_pi


class LinearSchedule(Schedule):
    """``alpha(t) = exp(-sigma(t))``, ``sigma(t) = sigma_min + t * (sigma_max - sigma_min)``."""

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sigma(t))

    def dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        return -(self.sigma_max - self.sigma_min) * self.alpha(t)


class GeometricSchedule(Schedule):
    """``alpha(t) = exp(-sigma(t))``, ``sigma(t) = sigma_min ** (1 - t) * sigma_max ** t``."""

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min ** (1.0 - t) * self.sigma_max**t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sigma(t))

    def dalpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(t)
        dsigma_dt = sigma * (math.log(self.sigma_max) - math.log(self.sigma_min))
        return -dsigma_dt * torch.exp(-sigma)
