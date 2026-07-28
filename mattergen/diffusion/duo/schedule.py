# Adapted from https://github.com/s-sahoo/duo, which is released under the
# Apache License, Version 2.0.

"""Continuous-time quantities specific to Duo, derived from a shared
``mattergen.diffusion.continuous_time.schedule.Schedule``.
"""

import torch

from mattergen.diffusion.continuous_time.schedule import Schedule


def duo_jump_rate(schedule: Schedule, t: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``lambda_Duo(t) = -alpha'(t) / (K * alpha(t))``."""
    alpha_t = schedule.alpha(t)
    dalpha_dt = schedule.dalpha_dt(t)
    return -dalpha_dt / (num_classes * alpha_t)
