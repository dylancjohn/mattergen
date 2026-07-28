"""test_charges.py

Unit tests for mattergen/constraints/charges.py.
"""

from __future__ import annotations

from neutral_layer.data.vocab import build_species_vocab

from mattergen.constraints.charges import build_charge_of


def test_charge_of_length_matches_vocab_plus_mask():
    vocab = build_species_vocab()
    charge_of, mask_idx = build_charge_of()

    assert len(charge_of) == vocab.num_species + 1
    assert mask_idx == vocab.num_species


def test_mask_entry_has_zero_charge():
    charge_of, mask_idx = build_charge_of()

    assert charge_of[mask_idx] == 0
    assert mask_idx == len(charge_of) - 1


def test_charge_of_matches_vocab_oxidation_states():
    vocab = build_species_vocab()
    charge_of, _ = build_charge_of()

    # 0-based index k in charge_of corresponds to 1-based species index k+1
    # in the vocab; every non-MASK entry must round-trip exactly.
    for k in range(vocab.num_species):
        assert charge_of[k] == vocab.oxidation_state_of(k + 1)


def test_charge_of_entries_are_ints():
    charge_of, _ = build_charge_of()

    assert all(isinstance(c, int) for c in charge_of)


def test_build_charge_of_is_deterministic():
    charge_of_1, mask_idx_1 = build_charge_of()
    charge_of_2, mask_idx_2 = build_charge_of()

    assert charge_of_1 == charge_of_2
    assert mask_idx_1 == mask_idx_2
