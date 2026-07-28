"""Tests for the unconstrained MDLM atom-type diffusion family.

Mirrors the structure of test_d3pm.py: schedule/marginal invariants, SUBS
parameterisation correctness, and an end-to-end loss sanity check with a
synthetic denoiser.
"""

from __future__ import annotations

import torch

from mattergen.diffusion.continuous_time.schedule import LogLinearSchedule
from mattergen.diffusion.corruption.mdlm_corruption import MDLMCorruption
from mattergen.diffusion.mdlm.mdlm_loss import mdlm_loss
from mattergen.diffusion.mdlm.schedule import mdlm_loss_weight, reveal_prob
from mattergen.diffusion.mdlm.subs import subs_log_probs, true_clean_state_log_prob

K = 5  # 4 clean species + MASK
MASK_IDX = K - 1


def _corruption(eps: float = 1e-3) -> MDLMCorruption:
    return MDLMCorruption(schedule=LogLinearSchedule(eps=eps), num_classes=K, offset=1)


# ---------------------------------------------------------------------------
# Schedule-derived quantities
# ---------------------------------------------------------------------------


def test_mdlm_loss_weight_matches_formula():
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    expected = -schedule.dalpha_dt(t) / (1.0 - schedule.alpha(t))
    assert torch.allclose(mdlm_loss_weight(schedule, t), expected)


def test_reveal_prob_matches_formula():
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.tensor([0.5, 0.8], dtype=torch.float64)
    r = torch.tensor([0.2, 0.3], dtype=torch.float64)
    expected = (schedule.alpha(r) - schedule.alpha(t)) / (1.0 - schedule.alpha(t))
    assert torch.allclose(reveal_prob(schedule, t, r), expected)


def test_reveal_prob_is_one_when_r_is_zero():
    # alpha(0) = 1, so u_{0,t} = (1 - alpha(t)) / (1 - alpha(t)) = 1: a full reveal.
    schedule = LogLinearSchedule(eps=1e-3)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    r = torch.zeros_like(t)
    assert torch.allclose(reveal_prob(schedule, t, r), torch.ones_like(t), atol=1e-6)


# ---------------------------------------------------------------------------
# MDLMCorruption
# ---------------------------------------------------------------------------


def test_prior_sampling_is_all_mask():
    corruption = _corruption()
    z = corruption.prior_sampling(shape=(100,))
    # 1-based (offset=1): MASK index K-1 becomes K-1+1 = K.
    assert torch.equal(z, torch.full((100,), MASK_IDX + 1, dtype=torch.long))


def test_offset_round_trips():
    corruption = _corruption()
    x = torch.tensor([1, 2, 3, 4, 5])
    assert torch.equal(corruption._to_non_zero_based(corruption._to_zero_based(x)), x)


def test_marginal_prob_returns_alpha_t():
    corruption = _corruption()
    t = torch.tensor([0.2, 0.6])
    batch_idx = torch.tensor([0, 0, 1])
    alpha_t, std = corruption.marginal_prob(x=torch.zeros(3), t=t, batch_idx=batch_idx)
    assert std is None
    expected = corruption.schedule.alpha(t)[batch_idx]
    assert torch.allclose(alpha_t, expected)


def test_sample_marginal_empirical_mask_frequency_matches_1_minus_alpha():
    torch.manual_seed(0)
    corruption = _corruption(eps=1e-3)
    n = 200_000
    x0 = torch.randint(0, K - 1, (n,)) + 1  # 1-based clean species, never MASK
    t = torch.full((n,), 0.7)
    xt = corruption.sample_marginal(x=x0, t=t, batch_idx=torch.arange(n))
    xt_zero = corruption._to_zero_based(xt)
    empirical_mask_freq = (xt_zero == MASK_IDX).float().mean().item()
    expected_mask_freq = 1.0 - corruption.schedule.alpha(torch.tensor(0.7)).item()
    assert abs(empirical_mask_freq - expected_mask_freq) < 5e-3


def test_sample_marginal_never_changes_clean_value_when_not_masked():
    torch.manual_seed(1)
    corruption = _corruption()
    n = 1000
    x0 = torch.randint(0, K - 1, (n,)) + 1
    t = torch.full((n,), 0.5)
    xt = corruption.sample_marginal(x=x0, t=t, batch_idx=torch.arange(n))
    xt_zero = corruption._to_zero_based(xt)
    x0_zero = corruption._to_zero_based(x0)
    revealed = xt_zero != MASK_IDX
    assert torch.equal(xt_zero[revealed], x0_zero[revealed])


def test_prior_logp_is_zero_for_all_mask_and_neg_inf_otherwise():
    corruption = _corruption()
    z = torch.full((4,), MASK_IDX + 1, dtype=torch.long)
    z[0] = 1  # not MASK
    batch_idx = torch.tensor([0, 0, 1, 1])
    logp = corruption.prior_logp(z, batch_idx=batch_idx)
    assert logp[0] == float("-inf")  # structure 0 has a non-mask atom
    assert logp[1] == 0.0  # structure 1 is fully masked


# ---------------------------------------------------------------------------
# SUBS parameterisation
# ---------------------------------------------------------------------------


def test_subs_visible_sites_get_exact_delta():
    raw_logits = torch.randn(3, K, dtype=torch.float64)
    xt_zero = torch.tensor([0, 2, MASK_IDX])
    log_probs = subs_log_probs(raw_logits, xt_zero, MASK_IDX)
    # Site 0 and 1 are visible: exact one-hot delta at the observed category.
    assert torch.allclose(log_probs[0, 0], torch.tensor(0.0, dtype=torch.float64))
    assert (log_probs[0, [1, 2, 3, 4]] < -1e5).all()
    assert torch.allclose(log_probs[1, 2], torch.tensor(0.0, dtype=torch.float64))


def test_subs_probabilities_normalise():
    raw_logits = torch.randn(4, K, dtype=torch.float64)
    xt_zero = torch.tensor([0, MASK_IDX, MASK_IDX, 3])
    log_probs = subs_log_probs(raw_logits, xt_zero, MASK_IDX)
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(4, dtype=torch.float64), atol=1e-6)


def test_subs_masked_sites_never_predict_mask():
    raw_logits = torch.randn(5, K, dtype=torch.float64) * 3
    xt_zero = torch.full((5,), MASK_IDX)
    log_probs = subs_log_probs(raw_logits, xt_zero, MASK_IDX)
    assert (log_probs[:, MASK_IDX].exp() < 1e-9).all()


def test_true_clean_state_log_prob_matches_gather():
    raw_logits = torch.randn(6, K, dtype=torch.float64)
    xt_zero = torch.tensor([0, MASK_IDX, 2, MASK_IDX, 1, MASK_IDX])
    x0_zero = torch.tensor([0, 1, 2, 3, 1, 0])
    direct = true_clean_state_log_prob(raw_logits, xt_zero, x0_zero, MASK_IDX)
    via_full = subs_log_probs(raw_logits, xt_zero, MASK_IDX).gather(
        -1, x0_zero.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(direct, via_full)


# ---------------------------------------------------------------------------
# End-to-end loss sanity
# ---------------------------------------------------------------------------


def test_mdlm_loss_zero_when_no_masked_sites():
    corruption = _corruption()
    batch_size = 2
    x0 = torch.tensor([1, 2, 3, 4])  # 1-based, never MASK
    xt = x0.clone()  # fully visible: no site masked
    batch_idx = torch.tensor([0, 0, 1, 1])
    t = torch.tensor([0.3, 0.7])
    score = torch.randn(4, K, dtype=torch.float64, requires_grad=True)

    loss = mdlm_loss(
        corruption=corruption,
        score_model_output=score,
        t=t,
        batch_idx=batch_idx,
        batch_size=batch_size,
        x=x0,
        noisy_x=xt,
        reduce="sum",
    )
    assert torch.allclose(loss, torch.zeros(batch_size, dtype=loss.dtype))


def test_mdlm_loss_matches_manual_computation_single_masked_atom():
    corruption = _corruption()
    x0 = torch.tensor([2])  # 1-based clean species index 2 -> 0-based 1
    xt = torch.tensor([MASK_IDX + 1])  # 1-based MASK
    batch_idx = torch.tensor([0])
    t = torch.tensor([0.4], dtype=torch.float64)
    score = torch.randn(1, K, dtype=torch.float64, requires_grad=True)

    loss = mdlm_loss(
        corruption=corruption,
        score_model_output=score,
        t=t,
        batch_idx=batch_idx,
        batch_size=1,
        x=x0,
        noisy_x=xt,
        reduce="sum",
    )

    x0_zero = torch.tensor([1])
    xt_zero = torch.tensor([MASK_IDX])
    log_p_true = true_clean_state_log_prob(score, xt_zero, x0_zero, MASK_IDX)
    lambda_t = mdlm_loss_weight(corruption.schedule, t)
    expected = -log_p_true * lambda_t

    assert torch.allclose(loss, expected)


def test_mdlm_loss_gradcheck():
    corruption = _corruption()
    x0 = torch.tensor([1, 2])
    xt = torch.tensor([MASK_IDX + 1, MASK_IDX + 1])
    batch_idx = torch.tensor([0, 1])
    t = torch.tensor([0.5, 0.5], dtype=torch.float64)
    score = torch.randn(2, K, dtype=torch.float64, requires_grad=True)

    def f(score_):
        return mdlm_loss(
            corruption=corruption,
            score_model_output=score_,
            t=t,
            batch_idx=batch_idx,
            batch_size=2,
            x=x0,
            noisy_x=xt,
            reduce="sum",
        )

    assert torch.autograd.gradcheck(f, (score,), eps=1e-6, atol=1e-4)
