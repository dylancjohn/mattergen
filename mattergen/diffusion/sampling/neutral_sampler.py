"""Structured (charge-neutral) clean-state sampling shared by the D3PM and MDLM predictors."""

from __future__ import annotations

import torch
from neutral_layer.generation.dp import (
    compute_q_max,
    flat_to_padded,
    neutral_sample,
    padded_to_flat,
)
from torch import FloatTensor, LongTensor, Tensor

from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.corruption.candidate_pinning import pin_committed_candidates


class NeutralSampler:
    """Joint charge-neutral clean-state sampler, applied as a logits transform.

    Draws one exact joint sample of ``x_0`` from the structured distribution
    (``neutral_layer.generation.dp.neutral_sample``) and returns it as one-hot
    logits. ``charge_of`` gives the integer charge per 0-based species-vocabulary
    index, ending with MASK (charge 0) at ``mask_idx``. Both default to
    :func:`build_charge_of` and must be given together or not at all.
    """

    def __init__(
        self,
        charge_of: list[int] | None = None,
        mask_idx: int | None = None,
    ) -> None:
        if (charge_of is None) != (mask_idx is None):
            raise ValueError(
                "charge_of and mask_idx must either both be supplied or both be omitted"
            )
        if charge_of is None:
            charge_of, mask_idx = build_charge_of()
        assert charge_of is not None and mask_idx is not None
        if not charge_of:
            raise ValueError("charge_of must contain at least the absorbing MASK class")
        if mask_idx != len(charge_of) - 1:
            raise ValueError(
                "absorbing D3PM requires MASK to be the final class; "
                f"got mask_idx={mask_idx}, K={len(charge_of)}"
            )
        if charge_of[mask_idx] != 0:
            raise ValueError("the absorbing MASK class must have charge zero")

        self.charge_of = list(charge_of)
        self.mask_idx = mask_idx
        self._charge_tensor_cache: dict[str, LongTensor] = {}

    def _get_charge_tensor(self, device: torch.device) -> LongTensor:
        key = str(device)
        if key not in self._charge_tensor_cache:
            self._charge_tensor_cache[key] = torch.tensor(
                self.charge_of, dtype=torch.long, device=device
            )
        return self._charge_tensor_cache[key]

    @torch.no_grad()
    def __call__(
        self,
        logits: FloatTensor,
        x_t: LongTensor,
        t: Tensor,
        batch_idx: LongTensor,
    ) -> FloatTensor:
        """Sample ``x_0`` jointly from the charge-neutral distribution.

        ``logits`` are the raw 0-based model logits for ``p_θ(x_0 | x_t)``,
        flat ``[N_atoms, K]``. ``x_t`` ``[N_atoms]`` holds the 1-based noisy
        indices as stored in ChemGraph (MASK is ``mask_idx + 1``); committed
        sites keep their value. ``t`` is unused and kept for API symmetry.
        Returns flat ``[N_atoms, K]`` one-hot logits (0 at the sampled species,
        ``-inf`` elsewhere). Raises ``RuntimeError`` if a crystal has no
        charge-neutral assignment.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N_atoms, K], got {logits.shape}")
        if logits.shape[-1] != len(self.charge_of):
            raise ValueError(
                f"charge_of has length {len(self.charge_of)}, but logits have "
                f"{logits.shape[-1]} classes"
            )
        if x_t.ndim != 1 or x_t.shape[0] != logits.shape[0]:
            raise ValueError(
                f"x_t must have shape [{logits.shape[0]}], got {x_t.shape}"
            )
        if batch_idx.ndim != 1 or batch_idx.shape[0] != logits.shape[0]:
            raise ValueError(
                f"batch_idx must have shape [{logits.shape[0]}], got {batch_idx.shape}"
            )
        if batch_idx.numel() == 0:
            raise ValueError("NeutralSampler requires at least one atom")
        if bool(((x_t < 1) | (x_t > len(self.charge_of))).any()):
            raise ValueError(
                f"1-based x_t values must lie in [1, {len(self.charge_of)}]"
            )

        device = logits.device
        batch_size = int(batch_idx.max().item()) + 1

        x_t_zero = x_t - 1

        logits_pad, attention_mask = flat_to_padded(logits, batch_idx, batch_size)
        x_t_pad, _ = flat_to_padded(
            x_t_zero,
            batch_idx,
            batch_size,
            fill_value=self.mask_idx,
            validate_batch_idx=False,
        )

        B, N, K = logits_pad.shape
        MASK_IDX = self.mask_idx

        committed = x_t_pad != MASK_IDX  # used again below for the error message
        pinned = pin_committed_candidates(logits_pad, x_t_pad, attention_mask, MASK_IDX)

        n_sites = attention_mask.sum(dim=1).long()
        charge_tensor = self._get_charge_tensor(device)
        q_max = compute_q_max(self.charge_of, int(n_sites.max().item()))

        samples, feasible = neutral_sample(pinned, charge_tensor, q_max, n_sites)

        # Fail rather than silently emit a non-neutral structure.
        if not bool(feasible.all()):
            bad = (~feasible).nonzero(as_tuple=True)[0].tolist()
            committed_charge = {
                b: int(
                    sum(
                        self.charge_of[int(idx)]
                        for idx in x_t_pad[b][committed[b]].tolist()
                    )
                )
                for b in bad
            }
            raise RuntimeError(
                "NeutralSampler: no charge-neutral assignment exists for "
                f"crystal(s) {bad} (committed charge sum {committed_charge}). "
                "This is unreachable in normal reverse sampling; check the "
                "species vocabulary, any externally fixed atoms, or numerical "
                "issues in the charge-state DP."
            )

        one_hot = torch.full((B, N, K), float("-inf"), device=device)
        one_hot.scatter_(2, samples.unsqueeze(2), 0.0)

        # Restore padding positions (downstream softmax would NaN on all-inf).
        padding = ~attention_mask
        if padding.any():
            one_hot[padding] = logits_pad[padding]

        return padded_to_flat(one_hot, batch_idx, batch_size, validate_batch_idx=False)
