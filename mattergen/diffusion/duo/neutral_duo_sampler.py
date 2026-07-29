"""neutral_duo_sampler.py

Inference-time charge-neutrality constraint for MatterGen's continuous-time
Duo.

Unlike the absorbing families, Duo's candidates are never pinned, so this
cannot reuse D3PM/MDLM's ``NeutralSampler``: the local weight here is the
forward-likelihood tilt, not a committed-site pin.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from neutral_layer.generation.dp import (
    compute_q_max,
    flat_to_padded,
    neutral_sample,
    padded_to_flat,
)
from torch import FloatTensor, LongTensor, Tensor

from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.corruption.duo_corruption import DuoCorruption
from mattergen.diffusion.corruption.sde_lib import ScoreFunction
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.duo.duo_predictors_correctors import DuoAncestralSamplingPredictor
from mattergen.diffusion.duo.neutral_duo_tilt import build_tilted_local_weights
from mattergen.diffusion.duo.posterior import usdm_posterior
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class NeutralDuoSampler:
    """Joint charge-neutral clean-state sampler under the forward-likelihood tilt.

    Draws a single joint sample from the tilted charge-neutral clean-state
    distribution ``omega_tilde`` via left-to-right autoregressive sampling
    conditioned on the running charge
    (``neutral_layer.generation.dp.neutral_sample``). Raises ``RuntimeError``
    if no charge-neutral assignment exists.

    Args:
        charge_of: Integer charge per 0-based vocab index. Defaults to
            :func:`build_charge_of` with the trailing MASK entry dropped
            (Duo has no MASK class).
    """

    def __init__(self, charge_of: Optional[list[int]] = None) -> None:
        if charge_of is None:
            full_charge_of, mask_idx = build_charge_of()
            charge_of = full_charge_of[:mask_idx]
        if not charge_of:
            raise ValueError("charge_of must be non-empty")
        self.charge_of = list(charge_of)
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
        current_state: LongTensor,
        alpha_u: Tensor,
        batch_idx: LongTensor,
    ) -> LongTensor:
        """Sample jointly from the tilted charge-neutral distribution.

        Args:
            logits: Raw model clean-category logits, flat ``[N_atoms, K]``,
                0-based.
            current_state: Category to tilt towards (the current noisy
                state), flat ``[N_atoms]``, 0-based.
            alpha_u: Survival probability at the tilt time, per structure,
                shape ``[B]``.
            batch_idx: Crystal index per atom, ``[N_atoms]``.

        Returns:
            LongTensor ``[N_atoms]``: jointly sampled 0-based clean species,
            charge-neutral by construction.
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N_atoms, K], got {logits.shape}")
        if logits.shape[-1] != len(self.charge_of):
            raise ValueError(
                f"charge_of has length {len(self.charge_of)}, but logits have "
                f"{logits.shape[-1]} classes"
            )
        if current_state.shape != (logits.shape[0],):
            raise ValueError(
                f"current_state must have shape [{logits.shape[0]}], got {current_state.shape}"
            )
        if batch_idx.shape != (logits.shape[0],):
            raise ValueError(
                f"batch_idx must have shape [{logits.shape[0]}], got {batch_idx.shape}"
            )
        if batch_idx.numel() == 0:
            raise ValueError("NeutralDuoSampler requires at least one atom")

        device = logits.device
        batch_size = int(batch_idx.max().item()) + 1
        if alpha_u.shape != (batch_size,):
            raise ValueError(f"alpha_u must have shape [{batch_size}], got {alpha_u.shape}")

        logits_pad, attention_mask = flat_to_padded(logits, batch_idx, batch_size)
        state_pad, _ = flat_to_padded(
            current_state, batch_idx, batch_size, fill_value=0, validate_batch_idx=False
        )

        tilted = build_tilted_local_weights(logits_pad, state_pad, alpha_u, len(self.charge_of))
        tilted = torch.where(
            attention_mask.unsqueeze(-1), tilted, torch.full_like(tilted, float("-inf"))
        )

        n_sites = attention_mask.sum(dim=1).long()
        charge_tensor = self._get_charge_tensor(device)
        q_max = compute_q_max(self.charge_of, int(n_sites.max().item()))

        samples_pad, feasible = neutral_sample(tilted, charge_tensor, q_max, n_sites)

        if not bool(feasible.all()):
            bad = (~feasible).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"NeutralDuoSampler: no charge-neutral assignment exists for crystal(s) "
                f"{bad}. Check the species vocabulary and model logits."
            )

        return padded_to_flat(samples_pad, batch_idx, batch_size, validate_batch_idx=False)


class NeutralDuoAncestralSamplingPredictor(DuoAncestralSamplingPredictor):
    """Duo ancestral predictor that enforces charge neutrality.

    For each reverse step: (1) form tilted local weights and draw one
    complete neutral clean assignment from ``omega_tilde`` via
    :class:`NeutralDuoSampler`; (2) conditional on that assignment, resample
    every site independently from the exact, unmodified uniform-state
    posterior (:func:`mattergen.diffusion.duo.posterior.usdm_posterior`),
    substituting a one-hot input in place of the raw softmax prediction.
    """

    def __init__(
        self,
        *,
        corruption: DuoCorruption,
        score_fn: ScoreFunction,
    ):
        super().__init__(corruption=corruption, score_fn=score_fn)
        self.neutral_sampler = NeutralDuoSampler()

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: Optional[BatchedData],
    ) -> SampleAndMean:
        assert isinstance(self.corruption, DuoCorruption)
        corruption = self.corruption

        xt_zero = corruption._to_zero_based(x.long())
        t_per_atom = maybe_expand(t, batch_idx)
        r_per_atom = (t_per_atom + dt).clamp(min=0.0)

        sampled_s0 = self.neutral_sampler(score, xt_zero, corruption.schedule.alpha(t), batch_idx)
        x0_probs = F.one_hot(sampled_s0, corruption.num_classes).to(score.dtype)

        alpha_t = corruption.schedule.alpha(t_per_atom)
        alpha_r = corruption.schedule.alpha(r_per_atom)
        posterior = usdm_posterior(x0_probs, xt_zero, alpha_r, alpha_t, corruption.num_classes)
        posterior = posterior.clamp(min=0.0)

        sample = torch.distributions.Categorical(probs=posterior).sample()
        expected = torch.argmax(posterior, dim=-1)

        return (
            corruption._to_non_zero_based(sample),
            corruption._to_non_zero_based(expected),
        )
