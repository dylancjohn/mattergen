"""Species-vocabulary to charge mapping for the charge-neutrality constraint."""

from __future__ import annotations


def build_charge_of() -> tuple[list[int], int]:
    """Return ``(charge_of, mask_idx)`` for the default species vocabulary.

    ``charge_of`` gives the integer oxidation state of each 0-based vocab
    index and has length ``num_species + 1``; its last entry is MASK, with
    charge 0, at ``mask_idx = num_species``.
    """
    from neutral_layer.data.vocab import build_species_vocab

    vocab = build_species_vocab()

    # 0-based index k corresponds to 1-based species index k+1.
    charge_of = [vocab.oxidation_state_of(i + 1) for i in range(vocab.num_species)]

    charge_of.append(0)
    mask_idx = vocab.num_species

    return charge_of, mask_idx
