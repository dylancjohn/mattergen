# Adapted from https://github.com/kuleshov-group/mdlm, which is released
# under the Apache License, Version 2.0.

"""Continuous-time quantities specific to MDLM, derived from a shared
``mattergen.diffusion.continuous_time.schedule.Schedule``.
"""

import torch

from mattergen.diffusion.continuous_time.schedule import Schedule


def mdlm_loss_weight(schedule: Schedule, t: torch.Tensor) -> torch.Tensor:
    """``lambda_MDLM(t) = -alpha'(t) / (1 - alpha(t))``."""
    alpha_t = schedule.alpha(t)
    dalpha_dt = schedule.dalpha_dt(t)
    return -dalpha_dt / (1.0 - alpha_t)


def reveal_prob(schedule: Schedule, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Probability that a site masked at time ``t`` is revealed by time ``r <= t``:
    ``u_{r,t} = (alpha(r) - alpha(t)) / (1 - alpha(t))``.
    """
    alpha_t = schedule.alpha(t)
    alpha_r = schedule.alpha(r)
    return (alpha_r - alpha_t) / (1.0 - alpha_t)
