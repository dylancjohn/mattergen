"""test_classifier_free_guidance.py

Tests for ``mattergen.diffusion.sampling.classifier_free_guidance``:

    * ``_combine_guided_scores`` matches plain ``torch.lerp`` away from -inf.
    * Entries excluded (-inf) in both the conditional and unconditional score
      stay -inf, for guidance scales other than 0/1, instead of becoming NaN.
    * A candidate excluded in only one branch still combines normally (no
      spurious -inf override).
"""

from __future__ import annotations

import torch

from mattergen.diffusion.sampling.classifier_free_guidance import _combine_guided_scores


def test_matches_lerp_for_finite_scores():
    unconditional = torch.tensor([0.1, -0.4, 2.0])
    conditional = torch.tensor([0.3, 0.2, -1.0])
    for scale in (0.3, 2.0, 5.0):
        combined = _combine_guided_scores(unconditional, conditional, scale)
        expected = torch.lerp(unconditional, conditional, scale)
        assert torch.allclose(combined, expected)


def test_jointly_excluded_entries_stay_neginf_not_nan():
    unconditional = torch.tensor([0.0, float("-inf"), 1.0])
    conditional = torch.tensor([0.0, float("-inf"), -2.0])
    for scale in (0.3, 1.0, 2.0, 5.0):
        combined = _combine_guided_scores(unconditional, conditional, scale)
        assert torch.isfinite(combined[0])
        assert torch.isfinite(combined[2])
        assert combined[1] == float("-inf")
        assert not torch.isnan(combined).any()


def test_singly_excluded_entry_matches_plain_lerp():
    """Only one branch excludes the candidate (not a joint exclusion): the
    both-excluded override must not fire, leaving plain lerp's own value."""
    unconditional = torch.tensor([-5.0])
    conditional = torch.tensor([float("-inf")])
    combined = _combine_guided_scores(unconditional, conditional, 2.0)
    expected = torch.lerp(unconditional, conditional, 2.0)
    assert torch.equal(combined, expected)
