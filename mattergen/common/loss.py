# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional

from mattergen.diffusion.d3pm.d3pm_loss import d3pm_loss
from mattergen.diffusion.losses import SummedFieldLoss, denoising_score_matching
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import FieldLoss
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss


class MaterialsLoss(SummedFieldLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        atomic_numbers_loss_partial: Optional[FieldLoss] = None,
        d3pm_hybrid_lambda: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            atomic_numbers_loss_partial: the atomic_numbers loss to use (a Hydra
                partial, so family-specific kwargs live in its own config block).
                Defaults to unconstrained D3PM.
            d3pm_hybrid_lambda: deprecated alias for
                ``atomic_numbers_loss_partial=partial(d3pm_loss, d3pm_hybrid_lambda=...)``,
                kept for older saved configs. Do not combine with
                ``atomic_numbers_loss_partial``.
        """
        model_targets = {"pos": ModelTarget.score_times_std, "cell": ModelTarget.score_times_std}
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss,
                reduce=reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")

            if atomic_numbers_loss_partial is not None:
                if d3pm_hybrid_lambda is not None:
                    raise ValueError(
                        "d3pm_hybrid_lambda is deprecated and only applies to the "
                        "default D3PM loss; do not combine it with "
                        "atomic_numbers_loss_partial."
                    )
            else:
                atomic_numbers_loss_partial = partial(
                    d3pm_loss, d3pm_hybrid_lambda=d3pm_hybrid_lambda or 0.0
                )
            loss_fns["atomic_numbers"] = partial(
                atomic_numbers_loss_partial,
                reduce=reduce,
            )
        self.reduce = reduce
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            model_targets=model_targets,
        )
