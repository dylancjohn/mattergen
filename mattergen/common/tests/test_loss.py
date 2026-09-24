"""Tests for how ``MaterialsLoss`` selects the atomic_numbers loss, including
the deprecated ``d3pm_hybrid_lambda`` kwarg used by older saved configs."""

from __future__ import annotations

import functools

import pytest

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.d3pm.d3pm_loss import d3pm_loss


def test_deprecated_d3pm_hybrid_lambda_still_constructs_default_d3pm_loss():
    """Shipped pretrained checkpoints pass a bare ``d3pm_hybrid_lambda``."""
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
