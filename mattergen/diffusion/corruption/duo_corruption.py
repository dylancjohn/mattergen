# Forward process adapted from https://github.com/s-sahoo/duo
# which is released under the Apache License, Version 2.0.

import math
from typing import Optional, Tuple, Union

import torch
from torch_scatter import scatter_add

from mattergen.diffusion.continuous_time.schedule import Schedule
from mattergen.diffusion.corruption.corruption import B, maybe_expand
from mattergen.diffusion.corruption.discrete_corruption import DiscreteCorruption
from mattergen.diffusion.data.batched_data import BatchedData


class DuoCorruption(DiscreteCorruption):
    """Duo continuous-time uniform-state corruption process.

    ``q(s_t = b | s_0 = a) = alpha(t) * 1[b=a] + (1 - alpha(t)) / K``: the
    clean category is retained with probability ``alpha(t)``, otherwise
    replaced by an independent uniform draw over all ``K = num_classes``
    categories. There is no MASK category.
    """

    def __init__(
        self,
        schedule: Schedule,
        num_classes: int,
        offset: int = 0,
    ):
        super().__init__(offset=offset)
        self.schedule = schedule
        self.num_classes = num_classes

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Survival probability ``alpha(t)``, broadcast per-node."""
        alpha_t = maybe_expand(self.schedule.alpha(t), batch_idx)
        return alpha_t, None

    def prior_sampling(
        self,
        shape: Union[torch.Size, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        """Independent uniform-categorical terminal state."""
        return self._to_non_zero_based(torch.randint(0, self.num_classes, tuple(shape)))

    def prior_logp(
        self,
        z: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Log-density of the uniform prior: ``-log(K)`` per atom, summed per structure."""
        z0 = self._to_zero_based(z.long())
        log_prob_per_node = torch.full_like(
            z0, -math.log(self.num_classes), dtype=torch.float32
        )
        return scatter_add(log_prob_per_node, batch_idx, dim=0)

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Sample ``x_t`` given ``x_0``: keep w.p. ``alpha(t)``, else replace with a
        uniform draw over all ``K`` categories (which may equal the clean value)."""
        alpha_t = self.marginal_prob(x=x, t=t, batch_idx=batch_idx, batch=batch)[0]
        x0 = self._to_zero_based(x.long())
        move = torch.rand_like(alpha_t) < (1.0 - alpha_t)
        uniform_draw = torch.randint(0, self.num_classes, x0.shape, device=x0.device)
        xt = torch.where(move, uniform_draw, x0)
        return self._to_non_zero_based(xt)
