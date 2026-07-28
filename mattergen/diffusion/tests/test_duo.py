"""Tests for the unconstrained Duo atom-type diffusion family.

Mirrors the structure of test_d3pm.py: forward-process invariants, posterior
correctness, and an end-to-end loss sanity check. Crucially, cross-checks the
closed-form Rao-Blackwellised f_Duo NELBO (duo_loss._duo_rate_nll) against a
naive, literal double-sum over the explicit reverse-rate generalised-KL
objective -- the primary correctness oracle for the algebraically-fused port.
"""

from __future__ import annotations

import math

import torch

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.duo_corruption import DuoCorruption
from mattergen.diffusion.duo.duo_loss import _duo_rate_nll, duo_loss
from mattergen.diffusion.duo.posterior import usdm_posterior
from mattergen.diffusion.duo.schedule import duo_jump_rate

K = 4


def _corruption(eps: float = 1e-3) -> DuoCorruption:
    return DuoCorruption(schedule=LogLinearSchedule(eps=eps), num_classes=K, offset=1)


def _naive_duo_rate_nll(p_theta, xt: int, x0: int, alpha_t: float, dalpha_t: float, k: int) -> float:
    """Literal double-sum over d_rate(rho*(b) || rho_theta(b)), b != xt."""
    lambda_t = -dalpha_t / (k * alpha_t)

    def rho_true(b):
        num = k * alpha_t * (1.0 if b == x0 else 0.0) + (1 - alpha_t)
        den = k * alpha_t * (1.0 if xt == x0 else 0.0) + (1 - alpha_t)
        return lambda_t * num / den

    def rho_theta(b):
        num = k * alpha_t * p_theta[b] + (1 - alpha_t)
        den = k * alpha_t * p_theta[xt] + (1 - alpha_t)
        return lambda_t * num / den

    total = 0.0
    for b in range(k):
        if b == xt:
            continue
        u = rho_true(b)
        v = rho_theta(b)
        total += u * math.log(u / v) - u + v
    return total


# ---------------------------------------------------------------------------
# Schedule-derived quantities
# ---------------------------------------------------------------------------


def test_duo_jump_rate_matches_formula():
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    expected = -schedule.dalpha_dt(t) / (K * schedule.alpha(t))
    assert torch.allclose(duo_jump_rate(schedule, t, K), expected)


# ---------------------------------------------------------------------------
# DuoCorruption
# ---------------------------------------------------------------------------


def test_prior_sampling_is_uniform_over_k_classes():
    torch.manual_seed(0)
    corruption = _corruption()
    z = corruption.prior_sampling(shape=(200_000,))
    z0 = corruption._to_zero_based(z)
    assert z0.min() >= 0 and z0.max() <= K - 1
    counts = torch.bincount(z0, minlength=K).float() / z0.numel()
    assert torch.allclose(counts, torch.full((K,), 1.0 / K), atol=0.01)


def test_prior_logp_is_uniform_log_density():
    corruption = _corruption()
    z = torch.randint(1, K + 1, (6,))
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])
    logp = corruption.prior_logp(z, batch_idx=batch_idx)
    expected = torch.full((2,), 3 * (-math.log(K)))
    assert torch.allclose(logp, expected)


def test_marginal_prob_returns_alpha_t():
    corruption = _corruption()
    t = torch.tensor([0.3, 0.6])
    batch_idx = torch.tensor([0, 0, 1])
    alpha_t, std = corruption.marginal_prob(x=torch.zeros(3), t=t, batch_idx=batch_idx)
    assert std is None
    assert torch.allclose(alpha_t, corruption.schedule.alpha(t)[batch_idx])


def test_sample_marginal_matches_forward_marginal_empirically():
    """P(x_t = x_0) should equal alpha_t + (1-alpha_t)/K -- the survival term
    plus the chance a uniform replacement lands back on x_0."""
    torch.manual_seed(1)
    corruption = _corruption()
    n = 300_000
    x0 = torch.randint(0, K, (n,)) + 1
    t = torch.full((n,), 0.6)
    xt = corruption.sample_marginal(x=x0, t=t, batch_idx=torch.arange(n))
    empirical_stay = (xt == x0).float().mean().item()
    alpha_t = corruption.schedule.alpha(torch.tensor(0.6)).item()
    expected_stay = alpha_t + (1 - alpha_t) / K
    assert abs(empirical_stay - expected_stay) < 5e-3


def test_sample_marginal_off_diagonal_is_uniform():
    """P(x_t = b | x_0 = a), b != a, should equal (1-alpha_t)/K for every b."""
    torch.manual_seed(2)
    corruption = _corruption()
    n = 400_000
    x0 = torch.ones(n, dtype=torch.long)  # fixed clean category (0-based 0, 1-based 1)
    t = torch.full((n,), 0.5)
    xt = corruption.sample_marginal(x=x0, t=t, batch_idx=torch.arange(n))
    xt_zero = corruption._to_zero_based(xt)
    alpha_t = corruption.schedule.alpha(torch.tensor(0.5)).item()
    expected_off_diag = (1 - alpha_t) / K
    for b in range(1, K):  # 0-based clean category is 0; check b=1,2,3
        freq = (xt_zero == b).float().mean().item()
        assert abs(freq - expected_off_diag) < 5e-3


# ---------------------------------------------------------------------------
# USDM posterior
# ---------------------------------------------------------------------------


def test_usdm_posterior_normalises():
    torch.manual_seed(3)
    n = 20
    x0_probs = torch.softmax(torch.randn(n, K, dtype=torch.float64), dim=-1)
    xt = torch.randint(0, K, (n,))
    alpha_t = torch.full((n,), 0.3, dtype=torch.float64)
    alpha_r = torch.full((n,), 0.7, dtype=torch.float64)
    posterior = usdm_posterior(x0_probs, xt, alpha_r, alpha_t, K)
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(n, dtype=torch.float64), atol=1e-8)


def test_usdm_posterior_matches_bayes_rule_with_true_one_hot_x0():
    """With a TRUE one-hot x0 (not the mean-parameterised simplex), the USDM
    posterior must equal the exact one-site forward posterior from Bayes' rule."""
    torch.manual_seed(4)
    a, c = 1, 2  # true clean category, current noisy category
    alpha_t, alpha_r = 0.3, 0.7
    x0_one_hot = torch.zeros(1, K, dtype=torch.float64)
    x0_one_hot[0, a] = 1.0
    xt = torch.tensor([c])

    posterior = usdm_posterior(
        x0_one_hot,
        xt,
        torch.tensor([alpha_r], dtype=torch.float64),
        torch.tensor([alpha_t], dtype=torch.float64),
        K,
    )[0]

    alpha_t_given_r = alpha_t / alpha_r
    denom = K * alpha_t * (1.0 if c == a else 0.0) + (1 - alpha_t)
    for b in range(K):
        num = (
            K * alpha_t * (1.0 if c == b == a else 0.0)
            + (alpha_t_given_r - alpha_t) * (1.0 if b == c else 0.0)
            + (alpha_r - alpha_t) * (1.0 if b == a else 0.0)
            + (1 - alpha_t_given_r) * (1 - alpha_r) / K
        )
        expected = num / denom
        assert abs(posterior[b].item() - expected) < 1e-8


def test_usdm_posterior_differs_from_generic_mixture():
    """The USDM plug-in is nonlinear: substituting a non-degenerate simplex
    x0_probs must NOT equal the linear mixture of one-hot posteriors weighted
    by x0_probs."""
    torch.manual_seed(5)
    xt = torch.tensor([1])
    alpha_t = torch.tensor([0.3], dtype=torch.float64)
    alpha_r = torch.tensor([0.7], dtype=torch.float64)
    x0_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)

    usdm = usdm_posterior(x0_probs, xt, alpha_r, alpha_t, K)[0]

    generic_mixture = torch.zeros(K, dtype=torch.float64)
    for a in range(K):
        one_hot = torch.zeros(1, K, dtype=torch.float64)
        one_hot[0, a] = 1.0
        generic_mixture += x0_probs[0, a] * usdm_posterior(one_hot, xt, alpha_r, alpha_t, K)[0]

    assert not torch.allclose(usdm, generic_mixture, atol=1e-4)


# ---------------------------------------------------------------------------
# f_Duo NELBO: agreement with the explicit rate objective (primary oracle)
# ---------------------------------------------------------------------------


def test_f_duo_matches_explicit_rate_objective():
    torch.manual_seed(6)
    schedule = LogLinearSchedule(eps=1e-3)
    for _ in range(20):
        t = torch.rand(1).item() * 0.9 + 0.05
        alpha_t = schedule.alpha(torch.tensor(t)).item()
        dalpha_t = schedule.dalpha_dt(torch.tensor(t)).item()
        x0, xt = torch.randint(0, K, (2,)).tolist()
        logits = torch.randn(K, dtype=torch.float64)
        p_theta = torch.softmax(logits, dim=-1)
        log_p_theta = torch.log_softmax(logits, dim=-1)

        closed_form = _duo_rate_nll(
            log_p_theta.unsqueeze(0),
            torch.tensor([xt]),
            torch.tensor([x0]),
            torch.tensor([alpha_t], dtype=torch.float64),
            torch.tensor([dalpha_t], dtype=torch.float64),
            K,
        ).item()
        naive = _naive_duo_rate_nll(p_theta.tolist(), xt, x0, alpha_t, dalpha_t, K)

        assert abs(closed_form - naive) < 1e-6, f"t={t}, x0={x0}, xt={xt}"


def test_f_duo_finite_and_gradcheck():
    torch.manual_seed(7)
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.tensor([0.4, 0.6], dtype=torch.float64)
    alpha_t = schedule.alpha(t)
    dalpha_t = schedule.dalpha_dt(t)
    xt = torch.tensor([0, 2])
    x0 = torch.tensor([1, 2])
    logits = torch.randn(2, K, dtype=torch.float64, requires_grad=True)

    def f(logits_):
        log_p = torch.log_softmax(logits_, dim=-1)
        return _duo_rate_nll(log_p, xt, x0, alpha_t, dalpha_t, K)

    loss = f(logits)
    assert torch.isfinite(loss).all()
    assert torch.autograd.gradcheck(f, (logits,), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end loss sanity
# ---------------------------------------------------------------------------


def test_duo_loss_finite_and_gradcheck():
    corruption = _corruption()
    x0 = torch.tensor([1, 2])
    xt = torch.tensor([2, 3])
    batch_idx = torch.tensor([0, 1])
    t = torch.tensor([0.4, 0.6], dtype=torch.float64)
    score = torch.randn(2, K, dtype=torch.float64, requires_grad=True)

    def f(score_):
        return duo_loss(
            corruption=corruption,
            score_model_output=score_,
            t=t,
            batch_idx=batch_idx,
            batch_size=2,
            x=x0,
            noisy_x=xt,
            reduce="sum",
        )

    loss = f(score)
    assert torch.isfinite(loss).all()
    assert torch.autograd.gradcheck(f, (score,), eps=1e-6, atol=1e-4)
