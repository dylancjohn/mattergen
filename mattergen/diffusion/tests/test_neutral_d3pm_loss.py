"""test_neutral_d3pm_loss.py

Tests for the structured charge-neutral training loss in
``mattergen.diffusion.d3pm.neutral_d3pm_loss``:

    * CE term (vb_weight=0): gradcheck w.r.t. the raw logits.
    * At t==0 the VB term reveals every site, so L_vb == L_ce (value + gradcheck).
    * Weight 0 short-circuits (returns zeros).
    * A neutral clean state gives a finite loss (feasible every step).
    * Intermediate-time gradients match exhaustive transition-weighted enumeration.
    * Invalid targets, partitions, vocabularies and MC configuration fail loudly.
    * make_neutral_d3pm_loss wires the constrained loss into MaterialsLoss's
      injectable atomic_numbers_loss_partial.
"""

from __future__ import annotations

import functools
import itertools
from pathlib import Path

import hydra
import pytest
import torch
from neutral_layer.generation.dp import compute_q_max, neutral_log_z
from omegaconf import OmegaConf

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.d3pm.neutral_d3pm_loss import (
    _numerator_logits,
    _pin_denominator_logits,
    make_neutral_d3pm_loss,
    neutral_d3pm_loss,
)

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


def test_materials_loss_wires_constrained_atom_loss():
    loss = MaterialsLoss(
        atomic_numbers_loss_partial=make_neutral_d3pm_loss(
            vb_weight=1.0, ce_weight=0.01, mc_samples=1
        )
    )
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert isinstance(atom_fn, functools.partial)
    assert atom_fn.func is neutral_d3pm_loss
    assert atom_fn.keywords["vb_weight"] == 1.0
    assert atom_fn.keywords["ce_weight"] == 0.01


def test_neutral_loss_hydra_config_instantiates():
    config_path = (
        Path(__file__).parents[2] / "conf/lightning_module/diffusion_module/neutral_d3pm.yaml"
    )
    config = OmegaConf.load(config_path)
    loss = hydra.utils.instantiate(config.loss_fn)
    assert isinstance(loss, MaterialsLoss)
    assert loss.loss_fns["atomic_numbers"].func is neutral_d3pm_loss


def test_t0_hybrid_ce_matches_base_mattergen_convention():
    """At reconstruction, MatterGen adds hybrid CE on top of the VB reconstruction."""
    score, x, noisy_x, batch_idx = _batch(seed=11)
    t = torch.zeros(2, dtype=torch.float64)
    vb_weight, ce_weight = 1.3, 0.2

    base = _call(score, x, noisy_x, batch_idx, t, vb=1.0, ce=0.0)
    combined = _call(
        score,
        x,
        noisy_x,
        batch_idx,
        t,
        vb=vb_weight,
        ce=ce_weight,
    )
    assert torch.allclose(combined, (vb_weight + ce_weight) * base, atol=1e-10)


def test_intermediate_reverse_gradient_matches_full_enumeration():
    """The absorbing support-only numerator has the exact structured gradient."""
    charges = CHARGE_OF
    mask_idx = MASK_IDX
    charge_tensor = torch.tensor(charges, dtype=torch.long)
    raw_logits = torch.tensor(
        [[0.8, -0.3, 0.2, -2.0], [-0.4, 0.5, 0.9, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    xt = torch.tensor([[mask_idx, mask_idx]], dtype=torch.long)
    s_prev = torch.tensor([[0, mask_idx]], dtype=torch.long)
    attention = torch.ones((1, 2), dtype=torch.bool)
    n_sites = torch.tensor([2], dtype=torch.long)
    q_max = compute_q_max(charges, 2)

    den_logits = _pin_denominator_logits(raw_logits.unsqueeze(0), xt, attention, mask_idx)
    num_logits = _numerator_logits(den_logits, xt, s_prev, attention, mask_idx)
    dp_loss = (
        neutral_log_z(den_logits, charge_tensor, q_max, n_sites)
        - neutral_log_z(num_logits, charge_tensor, q_max, n_sites)
    ).sum()
    dp_grad = torch.autograd.grad(dp_loss, raw_logits, retain_graph=True)[0]

    neutral_assignments = [
        assignment
        for assignment in itertools.product(range(mask_idx), repeat=2)
        if sum(charges[token] for token in assignment) == 0
    ]
    scores = torch.stack(
        [sum(raw_logits[i, token] for i, token in enumerate(a)) for a in neutral_assignments]
    )
    compatible = torch.tensor([a[0] == 0 for a in neutral_assignments])
    support_only_loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(
        scores[compatible], dim=0
    )
    support_grad = torch.autograd.grad(support_only_loss, raw_logits, retain_graph=True)[0]

    # Include the exact q(x_t | x_{t+1}, x_0) factors for the fixed reverse target.
    corruption = _corruption()
    log_q = []
    for assignment in neutral_assignments:
        q_prev, _, _ = corruption.d3pm.sample_and_compute_posterior_q(
            x_0=torch.tensor(assignment, dtype=torch.long),
            t=torch.full((2,), 4, dtype=torch.long),
            samples=xt[0],
            return_logits=False,
            return_transition_probs=True,
        )
        probability = q_prev[torch.arange(2), s_prev[0]].prod()
        log_q.append(probability.log())
    log_q_tensor = torch.stack(log_q).to(scores)
    full_loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(
        scores + log_q_tensor, dim=0
    )
    full_grad = torch.autograd.grad(full_loss, raw_logits)[0]

    assert torch.allclose(dp_grad, support_grad, atol=1e-10)
    assert torch.allclose(dp_grad, full_grad, atol=1e-10)
    assert not torch.allclose(dp_loss.detach(), full_loss.detach())


def test_committed_and_mask_columns_have_zero_gradient():
    score, x, noisy_x, batch_idx = _batch(seed=19)
    score.requires_grad_(True)
    loss = _call(
        score,
        x,
        noisy_x,
        batch_idx,
        torch.full((2,), 0.5, dtype=torch.float64),
        vb=0.0,
        ce=1.0,
    )
    loss.sum().backward()

    assert torch.equal(score.grad[[1, 3]], torch.zeros_like(score.grad[[1, 3]]))
    assert torch.equal(score.grad[:, MASK_IDX], torch.zeros_like(score.grad[:, MASK_IDX]))


def test_non_neutral_clean_target_raises():
    score, x, noisy_x, batch_idx = _batch()
    x = x.clone()
    x[1] = 2  # crystal 0 becomes charges (-1, 0)
    with pytest.raises(ValueError, match="charge-neutral clean targets"):
        _call(
            score,
            x,
            noisy_x,
            batch_idx,
            torch.full((2,), 0.5),
            vb=0.0,
            ce=1.0,
        )


def test_non_finite_denominator_raises():
    score, x, noisy_x, batch_idx = _batch()
    score = score.clone()
    score[0, :] = float("-inf")
    with pytest.raises(RuntimeError, match="non-finite denominator partition"):
        _call(
            score,
            x,
            noisy_x,
            batch_idx,
            torch.full((2,), 0.5),
            vb=0.0,
            ce=1.0,
        )


def test_non_finite_sampled_numerator_raises(monkeypatch):
    score, x, noisy_x, batch_idx = _batch()

    def _invalid_reverse_sample(self):
        return torch.zeros(self.probs.shape[0], dtype=torch.long, device=self.probs.device)

    monkeypatch.setattr(
        "mattergen.diffusion.d3pm.neutral_d3pm_loss.Categorical.sample",
        _invalid_reverse_sample,
    )
    with pytest.raises(RuntimeError, match="non-finite numerator"):
        _call(
            score,
            x,
            noisy_x,
            batch_idx,
            torch.full((2,), 0.5),
            vb=1.0,
            ce=0.0,
        )


@pytest.mark.parametrize("mc_samples", [0, -1, True, 1.5])
def test_invalid_mc_samples_raise(mc_samples):
    score, x, noisy_x, batch_idx = _batch()
    with pytest.raises(ValueError, match="positive integer"):
        _call(
            score,
            x,
            noisy_x,
            batch_idx,
            torch.full((2,), 0.5),
            vb=1.0,
            ce=0.0,
            mc=mc_samples,
        )
    with pytest.raises(ValueError, match="positive integer"):
        make_neutral_d3pm_loss(vb_weight=1.0, ce_weight=0.0, mc_samples=mc_samples)


def test_weight_zero_short_circuit_preserves_dtype():
    score, x, noisy_x, batch_idx = _batch(dtype=torch.float64)
    result = _call(
        score,
        x,
        noisy_x,
        batch_idx,
        torch.full((2,), 0.5, dtype=torch.float64),
        vb=0.0,
        ce=0.0,
    )
    assert result.dtype == score.dtype
