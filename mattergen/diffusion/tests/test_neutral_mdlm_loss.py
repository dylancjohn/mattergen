"""test_neutral_mdlm_loss.py

Tests for the structured charge-neutral training loss in
``mattergen.diffusion.mdlm.neutral_mdlm_loss``:

    * gradcheck w.r.t. the raw logits.
    * A neutral clean state gives a finite loss at every t.
    * Constrained one-site marginals match brute-force enumeration.
    * Committed and MASK-column gradients are exactly zero.
    * constraint-off (every candidate charge zero) reduces to plain mdlm_loss.
    * Invalid targets / partitions / vocabularies fail loudly.
    * make_neutral_mdlm_loss wires the constrained loss into MaterialsLoss's
      injectable atomic_numbers_loss_partial.
"""

from __future__ import annotations

import functools
import itertools
from pathlib import Path

import hydra
import pytest
import torch
from neutral_layer.generation.dp import compute_q_max, neutral_marginals_diff
from omegaconf import OmegaConf

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.candidate_pinning import pin_committed_candidates
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.mdlm.mdlm_loss import mdlm_loss
from mattergen.diffusion.mdlm.neutral_mdlm_loss import make_neutral_mdlm_loss, neutral_mdlm_loss

# Small synthetic charge vocab: species 0,1,2 (charges -1,0,+1) + MASK at index 3.
CHARGE_OF = [-1, 0, 1, 0]
MASK_IDX = 3
K = len(CHARGE_OF)


def _corruption() -> MDLMCorruption:
    return MDLMCorruption(schedule=LogLinearSchedule(), num_classes=K, offset=1)


def _batch(dtype=torch.float64, seed=0):
    """Two 2-atom neutral crystals; atom 0 masked, atom 1 committed.

    Clean 0-based species: crystal 0 = [0, 2] (charges -1,+1), crystal 1 = [2, 0].
    All 1-based externally (offset=1); MASK 1-based = MASK_IDX + 1 = 4.
    """
    torch.manual_seed(seed)
    x0_zero = torch.tensor([0, 2, 2, 0])
    x = x0_zero + 1
    xt_zero = torch.tensor([MASK_IDX, 2, MASK_IDX, 0])
    noisy_x = xt_zero + 1
    batch_idx = torch.tensor([0, 0, 1, 1])
    score = torch.randn(4, K, dtype=dtype)
    return score, x, noisy_x, batch_idx


def _call(score, x, noisy_x, batch_idx, t, *, reduce="sum", charge_of=CHARGE_OF):
    return neutral_mdlm_loss(
        corruption=_corruption(),
        score_model_output=score,
        t=t,
        batch_idx=batch_idx,
        batch_size=2,
        x=x,
        noisy_x=noisy_x,
        reduce=reduce,
        charge_of=charge_of,
        mask_idx=MASK_IDX,
    )


def test_gradcheck():
    score, x, noisy_x, batch_idx = _batch()
    score.requires_grad_(True)
    t = torch.tensor([0.5, 0.5], dtype=torch.float64)

    def f(s):
        return _call(s, x, noisy_x, batch_idx, t).sum()

    assert torch.autograd.gradcheck(f, (score,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_neutral_clean_state_is_feasible():
    score, x, noisy_x, batch_idx = _batch(dtype=torch.float32, seed=7)
    for tv in (0.05, 0.3, 0.9):
        t = torch.full((2,), tv)
        loss = _call(score, x, noisy_x, batch_idx, t)
        assert torch.isfinite(loss).all(), f"non-finite loss at t={tv}: {loss}"


def test_committed_and_mask_columns_have_zero_gradient():
    score, x, noisy_x, batch_idx = _batch(seed=19)
    score.requires_grad_(True)
    loss = _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5, dtype=torch.float64))
    loss.sum().backward()

    assert torch.equal(score.grad[[1, 3]], torch.zeros_like(score.grad[[1, 3]]))
    assert torch.equal(score.grad[:, MASK_IDX], torch.zeros_like(score.grad[:, MASK_IDX]))


def test_marginal_matches_brute_force_enumeration():
    """Constrained one-site marginals must match exhaustive
    enumeration over every neutral joint assignment. Uses two masked sites
    with no committed neighbour, so this genuinely exercises the joint DP,
    not just a single-site special case."""
    torch.manual_seed(2)
    raw_logits = torch.randn(2, K, dtype=torch.float64)
    xt = torch.tensor([[MASK_IDX, MASK_IDX]], dtype=torch.long)
    attention = torch.ones((1, 2), dtype=torch.bool)
    n_sites = torch.tensor([2], dtype=torch.long)
    q_max = compute_q_max(CHARGE_OF, 2)
    charge_tensor = torch.tensor(CHARGE_OF, dtype=torch.long)

    pinned = pin_committed_candidates(raw_logits.unsqueeze(0), xt, attention, MASK_IDX)
    log_marg, _ = neutral_marginals_diff(pinned, charge_tensor, q_max, n_sites)

    non_mask = list(range(MASK_IDX))
    assignments = [
        a
        for a in itertools.product(non_mask, repeat=2)
        if sum(CHARGE_OF[s] for s in a) == 0
    ]
    scores = torch.stack(
        [sum(raw_logits[i, s] for i, s in enumerate(a)) for a in assignments]
    )
    probs = torch.softmax(scores, dim=0)
    expected = torch.zeros(2, K, dtype=torch.float64)
    for a, p in zip(assignments, probs):
        for i, s in enumerate(a):
            expected[i, s] += p

    assert torch.allclose(log_marg[0].exp(), expected, atol=1e-10)


def test_constraint_off_matches_unconstrained_mdlm():
    """With every candidate charge zero, the neutrality constraint is
    vacuous (every joint assignment is trivially neutral), so the structured
    loss must reduce to plain masked cross-entropy."""
    zero_charge_of = [0, 0, 0, 0]
    score, x, noisy_x, batch_idx = _batch(dtype=torch.float64, seed=5)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)

    constrained = _call(score, x, noisy_x, batch_idx, t, charge_of=zero_charge_of)
    unconstrained = mdlm_loss(
        corruption=_corruption(),
        score_model_output=score,
        t=t,
        batch_idx=batch_idx,
        batch_size=2,
        x=x,
        noisy_x=noisy_x,
        reduce="sum",
    )
    assert torch.allclose(constrained, unconstrained, atol=1e-6)


def test_materials_loss_wires_constrained_atom_loss():
    loss = MaterialsLoss(atomic_numbers_loss_partial=make_neutral_mdlm_loss())
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert isinstance(atom_fn, functools.partial)
    assert atom_fn.func is neutral_mdlm_loss


def test_neutral_loss_hydra_config_instantiates():
    config_path = (
        Path(__file__).parents[2] / "conf/lightning_module/diffusion_module/neutral_mdlm.yaml"
    )
    config = OmegaConf.load(config_path)
    loss = hydra.utils.instantiate(config.loss_fn)
    assert isinstance(loss, MaterialsLoss)
    assert loss.loss_fns["atomic_numbers"].func is neutral_mdlm_loss


def test_non_neutral_clean_target_raises():
    score, x, noisy_x, batch_idx = _batch()
    x = x.clone()
    x[1] = 2  # crystal 0 becomes charges (-1, 0)
    with pytest.raises(ValueError, match="charge-neutral clean targets"):
        _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5))


def test_non_finite_partition_raises():
    score, x, noisy_x, batch_idx = _batch()
    score = score.clone()
    score[0, :] = float("-inf")
    with pytest.raises(RuntimeError, match="non-finite constrained partition"):
        _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5))


def test_invalid_vocab_raises():
    score, x, noisy_x, batch_idx = _batch()
    with pytest.raises(ValueError, match="MASK class must have charge zero"):
        _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5), charge_of=[-1, 0, 1, 5])
