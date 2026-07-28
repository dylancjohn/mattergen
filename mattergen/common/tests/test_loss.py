"""Tests for mattergen.common.loss.MaterialsLoss's loss-selection mechanism,
including backward compatibility with pre-reorganisation saved configs."""

from __future__ import annotations

import functools

import pytest

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.d3pm.d3pm_loss import d3pm_loss


def test_deprecated_d3pm_hybrid_lambda_still_constructs_default_d3pm_loss():
    """Older saved configs (e.g. shipped pretrained checkpoints) construct
    MaterialsLoss with a bare d3pm_hybrid_lambda kwarg; this must keep working."""
    loss = MaterialsLoss(d3pm_hybrid_lambda=0.01, include_atomic_numbers=True)
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert isinstance(atom_fn, functools.partial)
    assert atom_fn.func is d3pm_loss
    assert atom_fn.keywords["d3pm_hybrid_lambda"] == 0.01


def test_d3pm_hybrid_lambda_omitted_defaults_to_zero():
    loss = MaterialsLoss(include_atomic_numbers=True)
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert atom_fn.func is d3pm_loss
    assert atom_fn.keywords["d3pm_hybrid_lambda"] == 0.0


def test_atomic_numbers_loss_partial_takes_precedence_when_given():
    sentinel = functools.partial(d3pm_loss, d3pm_hybrid_lambda=0.5)
    loss = MaterialsLoss(atomic_numbers_loss_partial=sentinel, include_atomic_numbers=True)
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert atom_fn.func is d3pm_loss
    assert atom_fn.keywords["d3pm_hybrid_lambda"] == 0.5


def test_combining_both_args_raises():
    with pytest.raises(ValueError, match="deprecated"):
        MaterialsLoss(
            atomic_numbers_loss_partial=d3pm_loss,
            d3pm_hybrid_lambda=0.01,
            include_atomic_numbers=True,
        )
