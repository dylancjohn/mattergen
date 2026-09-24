"""Candidate pinning for absorbing-mask structured losses and samplers.

Committed sites collapse to a one-hot logit at their known value; the MASK
column and padding are always excluded. Used by D3PM and MDLM, which are both
absorbing-mask processes. Duo is uniform-state, so its sites are never
committed and it does not use this module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor


def pin_committed_candidates(
    logits_pad: FloatTensor,
    xt_pad: LongTensor,
    attention_mask: BoolTensor,
    mask_idx: int,
    *,
    candidate_mask: Optional[BoolTensor] = None,
) -> FloatTensor:
    """Collapse committed sites to one-hot; exclude MASK and padding.

    ``logits_pad`` ``[B, N, K]`` holds per-site log-scores and ``xt_pad``
    ``[B, N]`` the 0-based noisy indices, with padding already filled with
    ``mask_idx`` by the caller. Committed sites (``xt_pad != mask_idx``) become
    one-hot at their ``xt_pad`` value; the MASK column and padding (where
    ``attention_mask`` is ``False``) become ``-inf``. Other real sites keep
    their raw logits, so gradients flow only there. At those sites, ``False``
    entries of the optional ``candidate_mask`` ``[B, N, K]`` are also set to
    ``-inf``. Returns pinned logits ``[B, N, K]``.
    """
    K = logits_pad.shape[-1]
    device = logits_pad.device

    committed = xt_pad != mask_idx  # [B, N]
    committed_one_hot = torch.full_like(logits_pad, float("-inf"))
    committed_one_hot.scatter_(-1, xt_pad.clamp(min=0).unsqueeze(-1), 0.0)
    pinned = torch.where(committed.unsqueeze(-1), committed_one_hot, logits_pad)

    if candidate_mask is not None:
        exclude = ~candidate_mask & ~committed.unsqueeze(-1)
        pinned = torch.where(exclude, torch.full_like(pinned, float("-inf")), pinned)

    mask_col = F.one_hot(torch.tensor(mask_idx, device=device), K).bool()
    pinned = torch.where(mask_col, torch.full_like(pinned, float("-inf")), pinned)

    pinned = torch.where(
        attention_mask.unsqueeze(-1), pinned, torch.full_like(pinned, float("-inf"))
    )
    return pinned
