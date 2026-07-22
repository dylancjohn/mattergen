"""test_neutral_d3pm_loss.py

Tests for the structured charge-neutral training loss in
``mattergen.constraints.neutral_d3pm_loss``:

    * CE term (vb_weight=0): gradcheck w.r.t. the raw logits.
    * At t==0 the VB term reveals every site, so L_vb == L_ce (value + gradcheck).
    * Weight 0 short-circuits (returns zeros).
    * A neutral clean state gives a finite loss (feasible every step).
    * NeutralMaterialsLoss wires the constrained loss into the atomic_numbers field.
"""

from __future__ import annotations

import functools

import torch

from mattergen.constraints.neutral_d3pm_loss import (
    NeutralMaterialsLoss,
    neutral_d3pm_loss,
)
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule

# Small synthetic charge vocab: species 0,1,2 (charges -1,0,+1) + MASK at index 3.
CHARGE_OF = [-1, 0, 1, 0]
MASK_IDX = 3
K = len(CHARGE_OF)


def _corruption() -> D3PMCorruption:
    schedule = create_discrete_diffusion_schedule(
        kind="linear", beta_min=1e-3, beta_max=0.1, num_steps=10
    )
    return D3PMCorruption(d3pm=MaskDiffusion(dim=K, schedule=schedule), offset=1)


def _batch(dtype=torch.float64, seed=0):
    """Two 2-atom neutral crystals; atom 0 masked, atom 1 committed.

    Clean 0-based species: crystal 0 = [0, 2] (charges -1,+1), crystal 1 = [2, 0].
    All 1-based externally (offset=1); MASK 1-based = MASK_IDX + 1 = 4.
    """
    torch.manual_seed(seed)
    x0_zero = torch.tensor([0, 2, 2, 0])
    x = x0_zero + 1  # 1-based clean
    # noisy: atom 0 of each crystal masked, atom 1 committed (== clean).
    xt_zero = torch.tensor([MASK_IDX, 2, MASK_IDX, 0])
    noisy_x = xt_zero + 1  # 1-based (MASK → 4)
    batch_idx = torch.tensor([0, 0, 1, 1])
    score = torch.randn(4, K, dtype=dtype)
    return score, x, noisy_x, batch_idx


def _call(score, x, noisy_x, batch_idx, t, *, vb, ce, mc=1, reduce="sum"):
    return neutral_d3pm_loss(
        corruption=_corruption(),
        score_model_output=score,
        t=t,
        batch_idx=batch_idx,
        batch_size=2,
        x=x,
        noisy_x=noisy_x,
        reduce=reduce,
        vb_weight=vb,
        ce_weight=ce,
        mc_samples=mc,
        charge_of=CHARGE_OF,
        mask_idx=MASK_IDX,
    )


def test_ce_gradcheck():
    score, x, noisy_x, batch_idx = _batch()
    score.requires_grad_(True)
    t = torch.tensor([0.5, 0.5], dtype=torch.float64)

    def f(s):
        return _call(s, x, noisy_x, batch_idx, t, vb=0.0, ce=1.0).sum()

    assert torch.autograd.gradcheck(f, (score,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_vb_equals_ce_at_t0():
    score, x, noisy_x, batch_idx = _batch()
    t = torch.zeros(2, dtype=torch.float64)  # discrete t == 0 → reconstruction

    l_ce = _call(score, x, noisy_x, batch_idx, t, vb=0.0, ce=1.0)
    l_vb = _call(score, x, noisy_x, batch_idx, t, vb=1.0, ce=0.0)
    assert torch.allclose(l_ce, l_vb, atol=1e-5), f"L_ce={l_ce}, L_vb={l_vb}"


def test_vb_gradcheck_at_t0():
    score, x, noisy_x, batch_idx = _batch(seed=3)
    score.requires_grad_(True)
    t = torch.zeros(2, dtype=torch.float64)

    def f(s):
        return _call(s, x, noisy_x, batch_idx, t, vb=1.0, ce=0.0).sum()

    assert torch.autograd.gradcheck(f, (score,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_weight_zero_short_circuits():
    score, x, noisy_x, batch_idx = _batch()
    t = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = _call(score, x, noisy_x, batch_idx, t, vb=0.0, ce=0.0)
    assert torch.equal(loss, torch.zeros(2, dtype=torch.float64))


def test_neutral_clean_state_is_feasible():
    """A charge-neutral clean state must give a finite loss at every step."""
    score, x, noisy_x, batch_idx = _batch(dtype=torch.float32, seed=7)
    for tv in (0.0, 0.3, 0.9):
        t = torch.full((2,), tv)
        loss = _call(score, x, noisy_x, batch_idx, t, vb=1.0, ce=1.0, mc=2)
        assert torch.isfinite(loss).all(), f"non-finite loss at t={tv}: {loss}"


def test_vb_is_stochastic_but_finite_at_intermediate_t():
    score, x, noisy_x, batch_idx = _batch(seed=1)
    score.requires_grad_(True)
    t = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss = _call(score, x, noisy_x, batch_idx, t, vb=1.0, ce=0.0)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert score.grad is not None and torch.isfinite(score.grad).all()


def test_neutral_materials_loss_wires_constrained_atom_loss():
    loss = NeutralMaterialsLoss(vb_weight=1.0, ce_weight=0.01, mc_samples=1)
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert isinstance(atom_fn, functools.partial)
    assert atom_fn.func is neutral_d3pm_loss
    assert atom_fn.keywords["vb_weight"] == 1.0
    assert atom_fn.keywords["ce_weight"] == 0.01
