# Forward process adapted from https://github.com/kuleshov-group/mdlm
# which is released under the Apache License, Version 2.0.

from typing import Optional, Tuple, Union

import torch
from torch_scatter import scatter_add

from mattergen.diffusion.continuous_time.schedule import Schedule
from mattergen.diffusion.corruption.corruption import B, maybe_expand
from mattergen.diffusion.corruption.discrete_corruption import DiscreteCorruption
from mattergen.diffusion.data.batched_data import BatchedData


class MDLMCorruption(DiscreteCorruption):
    """MDLM continuous-time absorbing-mask corruption process.

    Each valid site retains its clean category with probability ``alpha(t)``
    and is otherwise replaced with the MASK token, the last of ``num_classes``
    categories.
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
        self.mask_index = num_classes - 1

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Survival probability ``alpha(t)`` for ``q(x_t = x_0 | x_0)``, broadcast per-node."""
        alpha_t = maybe_expand(self.schedule.alpha(t), batch_idx)
        return alpha_t, None

    def prior_sampling(
        self,
        shape: Union[torch.Size, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        """All-mask terminal state."""
        return self._to_non_zero_based(torch.full(shape, self.mask_index, dtype=torch.long))

    def prior_logp(
        self,
        z: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Log-density of the all-mask prior: 0 if masked, -inf otherwise."""
        z0 = self._to_zero_based(z.long())
        is_mask = z0 == self.mask_index
        log_prob_per_node = torch.where(
            is_mask,
            torch.zeros_like(z0, dtype=torch.float32),
            torch.full_like(z0, float("-inf"), dtype=torch.float32),
        )
        return scatter_add(log_prob_per_node, batch_idx, dim=0)

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Sample ``x_t`` given ``x_0``: keep the clean category w.p. ``alpha(t)``, else MASK."""
        alpha_t = self.marginal_prob(x=x, t=t, batch_idx=batch_idx, batch=batch)[0]
        x0 = self._to_zero_based(x.long())
        keep = torch.rand_like(alpha_t) < alpha_t
        xt = torch.where(keep, x0, torch.full_like(x0, self.mask_index))
        return self._to_non_zero_based(xt)
