"""candidate_pinning.py

Shared absorbing-mask candidate pinning for charge-neutral constrained losses
and samplers.  Committed sites collapse to a one-hot logit at their known
value; the MASK column and padding are always excluded.  D3PM's and MDLM's
constrained code both use this convention, since both are absorbing-mask
processes; Duo's is not (its candidates are never pinned), so it does not use
this module.
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

    Committed sites (``xt_pad != mask_idx``) are pinned to a one-hot logit at
    their value in ``xt_pad``; the MASK column and padding are set to
    ``-inf``; other real sites keep their raw ``logits_pad`` so gradients
    flow only there.

    Args:
        logits_pad: Per-site log-scores, shape ``[B, N, K]``.
        xt_pad: Per-site index tensor, shape ``[B, N]``, used both to decide
            which sites are committed (``xt_pad != mask_idx``) and the value
            to pin them to. Padding positions should already be filled with
            ``mask_idx`` by the caller.
        attention_mask: ``True`` at real (non-padding) sites, shape ``[B, N]``.
        mask_idx: Index of the MASK class within ``K``; always excluded from
            the result.
        candidate_mask: Optional boolean mask, shape ``[B, N, K]``. At
            non-committed real sites, ``False`` entries are excluded
            (``-inf``) on top of the pinning above.

    Returns:
        Pinned logits, shape ``[B, N, K]``.
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
