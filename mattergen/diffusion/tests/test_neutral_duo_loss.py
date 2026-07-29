"""test_neutral_duo_loss.py

Tests for the structured charge-neutral training loss in
``mattergen.diffusion.duo.neutral_duo_loss``:

    * gradcheck w.r.t. the raw logits.
    * A neutral clean state gives a finite loss at every t, including near
      the production min_t=1e-3 floor (the A_t/R_t -> inf numerical caution).
    * The marginal-shortcut rate ratio v_{i,b} matches a from-scratch
      brute-force enumeration of the singly-modified tilted partition ratio.
    * constraint-off (every candidate charge zero) reduces to plain duo_loss.
    * Invalid targets / partitions / vocabularies fail loudly.
    * make_neutral_duo_loss wires the constrained loss into MaterialsLoss's
      injectable atomic_numbers_loss_partial, with the MASK entry dropped.
"""

from __future__ import annotations

import functools
import itertools
import math
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from mattergen.common.loss import MaterialsLoss
from mattergen.constraints.charges import build_charge_of
from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.duo_corruption import DuoCorruption
from mattergen.diffusion.duo.duo_loss import duo_loss
from mattergen.diffusion.duo.neutral_duo_loss import make_neutral_duo_loss, neutral_duo_loss

# Small synthetic charge vocab: species 0,1,2 with charges -1,0,+1. No MASK class.
CHARGE_OF = [-1, 0, 1]
K = len(CHARGE_OF)


def _corruption() -> DuoCorruption:
    return DuoCorruption(schedule=LogLinearSchedule(), num_classes=K, offset=1)


def _batch(dtype=torch.float64, seed=0):
    """Two 2-atom neutral crystals, current (noisy) state given explicitly.

    Clean 0-based species: crystal 0 = [0, 2], crystal 1 = [2, 0].
    Current 0-based: crystal 0 = [1, 2] (site 0 differs from clean), crystal
    1 = [2, 0] (both sites already equal clean, by chance of the forward
    process -- Duo has no visible/masked distinction).
    """
    torch.manual_seed(seed)
    x0_zero = torch.tensor([0, 2, 2, 0])
    x = x0_zero + 1
    xt_zero = torch.tensor([1, 2, 2, 0])
    noisy_x = xt_zero + 1
    batch_idx = torch.tensor([0, 0, 1, 1])
    score = torch.randn(4, K, dtype=dtype)
    return score, x, noisy_x, batch_idx


def _call(score, x, noisy_x, batch_idx, t, *, reduce="sum", charge_of=CHARGE_OF):
    return neutral_duo_loss(
        corruption=_corruption(),
        score_model_output=score,
        t=t,
        batch_idx=batch_idx,
        batch_size=2,
        x=x,
        noisy_x=noisy_x,
        reduce=reduce,
        charge_of=charge_of,
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


def test_finite_and_gradient_finite_near_min_t_floor():
    """The A_t/R_t -> inf singularity as t -> 0 must stay controlled at the
    actual production floor (min_t=1e-3, LogLinearSchedule)."""
    score, x, noisy_x, batch_idx = _batch(dtype=torch.float64, seed=13)
    score.requires_grad_(True)
    t = torch.full((2,), 1e-3, dtype=torch.float64)

    loss = _call(score, x, noisy_x, batch_idx, t)
    assert torch.isfinite(loss).all(), f"non-finite loss at t=min_t: {loss}"
    loss.sum().backward()
    assert torch.isfinite(score.grad).all(), f"non-finite gradient at t=min_t: {score.grad}"


def test_v_stays_finite_at_production_scale_float32_min_t():
    """Random logits at K=428/float32/t=min_t never pushed v negative, but
    confident (peaked) predictions do -- the realistic case for a partially
    trained model. v is analytically >= R_t/(A_t+R_t) > 0, but confirmed
    empirically to go slightly negative here from float32 rounding before the
    non-negative-mixture reformulation; this pins that regression at the
    actual production scale, and cross-checks both loss value and gradient
    against the same computation done in float64 (a higher-precision
    reference the original cancellation-prone form could not agree with)."""
    K = 428
    charge_of = [(-1) ** i for i in range(K)]  # alternating -1/+1; (0,1) is a neutral pair
    corruption = DuoCorruption(schedule=LogLinearSchedule(), num_classes=K, offset=1)

    torch.manual_seed(0)
    n_atoms = 2
    dominant = torch.randint(0, K, (n_atoms,))  # peaked prediction, unrelated to the true target
    x = torch.tensor([0, 1], dtype=torch.long) + 1  # 1-based neutral clean target
    noisy_x = dominant.clone() + 1
    batch_idx = torch.zeros(n_atoms, dtype=torch.long)

    def _loss_and_grad(dtype):
        score = torch.zeros(n_atoms, K, dtype=dtype)
        score.scatter_(-1, dominant.unsqueeze(-1), 30.0)  # near-delta prediction
        score.requires_grad_(True)
        loss = neutral_duo_loss(
            corruption=corruption,
            score_model_output=score,
            t=torch.full((1,), 1e-3, dtype=dtype),
            batch_idx=batch_idx,
            batch_size=1,
            x=x,
            noisy_x=noisy_x,
            reduce="sum",
            charge_of=charge_of,
        )
        loss.sum().backward()
        return loss.detach(), score.grad.detach()

    loss32, grad32 = _loss_and_grad(torch.float32)
    loss64, grad64 = _loss_and_grad(torch.float64)
    assert torch.isfinite(loss32).all(), f"non-finite loss at K=428, t=min_t, float32: {loss32}"
    assert torch.isfinite(grad32).all(), f"non-finite gradient at K=428, t=min_t, float32"
    assert loss32.item() == pytest.approx(loss64.item(), rel=1e-3), (
        f"float32 loss {loss32.item()} disagrees with float64 reference {loss64.item()}"
    )
    assert torch.allclose(grad32.double(), grad64, atol=1e-3, rtol=1e-2), (
        f"float32 gradient disagrees with float64 reference: "
        f"max diff {(grad32.double() - grad64).abs().max().item()}"
    )


def test_hard_excluded_candidate_gets_zero_tilted_marginal():
    """A candidate excluded via true -inf (as mask_disallowed_species now
    produces) must get exactly zero probability in the tilted marginals, not
    just a small one -- exact support, not just low likelihood."""
    from neutral_layer.generation.dp import compute_q_max, neutral_marginals_diff

    from mattergen.diffusion.duo.neutral_duo_tilt import build_tilted_local_weights

    torch.manual_seed(7)
    N = 2
    raw_logits = torch.randn(1, N, K, dtype=torch.float64)
    raw_logits[0, 0, 1] = float("-inf")  # exclude species 1 at site 0
    xt = torch.zeros(1, N, dtype=torch.long)
    alpha_t = torch.tensor([0.6], dtype=torch.float64)

    tilted = build_tilted_local_weights(raw_logits, xt, alpha_t, K)
    charge_tensor = torch.tensor(CHARGE_OF, dtype=torch.long)
    q_max = compute_q_max(CHARGE_OF, N)
    n_sites = torch.tensor([N], dtype=torch.long)
    log_marg, log_z = neutral_marginals_diff(tilted, charge_tensor, q_max, n_sites)
    assert log_z.isfinite().all()
    assert log_marg[0, 0, 1].exp().item() == pytest.approx(0.0, abs=1e-12)


def test_v_matches_brute_force_modified_partition_ratio():
    """The marginal-shortcut rate ratio v_{i,b} must equal a from-scratch
    enumeration of Z_{theta,t,i}(b;s_t) / Z_{theta,t}(s_t)."""
    from neutral_layer.generation.dp import compute_q_max, neutral_marginals_diff

    from mattergen.diffusion.duo.neutral_duo_tilt import build_tilted_local_weights

    torch.manual_seed(4)
    N = 2
    raw_logits = torch.randn(N, K, dtype=torch.float64)
    xt = [1, 2]  # current state per site, 0-based
    alpha_t = 0.6

    def g(y, a):
        return alpha_t * (1.0 if y == a else 0.0) + (1 - alpha_t) / K

    neutral_assignments = [
        a for a in itertools.product(range(K), repeat=N) if sum(CHARGE_OF[s] for s in a) == 0
    ]

    def score(assignment, tilt_site=None, tilt_val=None):
        total = 0.0
        for i, s0 in enumerate(assignment):
            tilt_target = tilt_val if tilt_site == i else xt[i]
            total += raw_logits[i, s0].item() + math.log(g(tilt_target, s0))
        return total

    denom = torch.logsumexp(
        torch.tensor([score(a) for a in neutral_assignments]), dim=0
    )

    xt_tensor = torch.tensor([xt])
    charge_tensor = torch.tensor(CHARGE_OF, dtype=torch.long)
    q_max = compute_q_max(CHARGE_OF, N)
    n_sites = torch.tensor([N], dtype=torch.long)
    tilted = build_tilted_local_weights(
        raw_logits.unsqueeze(0), xt_tensor, torch.tensor([alpha_t], dtype=torch.float64), K
    )
    log_marg, log_z = neutral_marginals_diff(tilted, charge_tensor, q_max, n_sites)
    assert log_z.isfinite().all()
    omega = log_marg[0].exp()  # [N, K]

    A_t = K * alpha_t
    R_t = 1 - alpha_t
    for site_i in range(N):
        for b in range(K):
            if b == xt[site_i]:
                continue
            numer = torch.logsumexp(
                torch.tensor(
                    [score(a, tilt_site=site_i, tilt_val=b) for a in neutral_assignments]
                ),
                dim=0,
            )
            expected_ratio = (numer - denom).exp().item()

            omega_b = omega[site_i, b].item()
            omega_c = omega[site_i, xt[site_i]].item()
            actual_ratio = 1 + (A_t / R_t) * omega_b - (A_t / (A_t + R_t)) * omega_c

            assert actual_ratio == pytest.approx(expected_ratio, abs=1e-6), (
                f"site {site_i}, b={b}: shortcut={actual_ratio}, brute-force={expected_ratio}"
            )


def test_constraint_off_matches_unconstrained_duo():
    """With every candidate charge zero, the neutrality constraint is
    vacuous, so the structured loss must reduce to plain duo_loss."""
    zero_charge_of = [0, 0, 0]
    score, x, noisy_x, batch_idx = _batch(dtype=torch.float64, seed=5)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)

    constrained = _call(score, x, noisy_x, batch_idx, t, charge_of=zero_charge_of)
    unconstrained = duo_loss(
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
    loss = MaterialsLoss(atomic_numbers_loss_partial=make_neutral_duo_loss())
    atom_fn = loss.loss_fns["atomic_numbers"]
    assert isinstance(atom_fn, functools.partial)
    assert atom_fn.func is neutral_duo_loss


def test_make_neutral_duo_loss_drops_mask_entry():
    full_charge_of, mask_idx = build_charge_of()
    bound_loss = make_neutral_duo_loss()
    assert len(bound_loss.keywords["charge_of"]) == mask_idx
    assert bound_loss.keywords["charge_of"] == full_charge_of[:mask_idx]


def test_neutral_loss_hydra_config_instantiates():
    config_path = (
        Path(__file__).parents[2] / "conf/lightning_module/diffusion_module/neutral_duo.yaml"
    )
    config = OmegaConf.load(config_path)
    loss = hydra.utils.instantiate(config.loss_fn)
    assert isinstance(loss, MaterialsLoss)
    assert loss.loss_fns["atomic_numbers"].func is neutral_duo_loss


def test_non_neutral_clean_target_raises():
    score, x, noisy_x, batch_idx = _batch()
    x = x.clone()
    x[1] = 1  # crystal 0 becomes charges (-1, 0)
    with pytest.raises(ValueError, match="charge-neutral clean targets"):
        _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5))


def test_non_finite_partition_raises():
    score, x, noisy_x, batch_idx = _batch()
    score = score.clone()
    score[0, :] = float("-inf")
    with pytest.raises(RuntimeError, match="non-finite tilted partition"):
        _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5))


def test_vocab_length_mismatch_raises():
    score, x, noisy_x, batch_idx = _batch()
    with pytest.raises(ValueError, match="charge_of has length"):
        _call(score, x, noisy_x, batch_idx, torch.full((2,), 0.5), charge_of=[-1, 0])
