"""charges.py

Vocabulary → charge mapping for the charge-neutrality constraint.
"""

from __future__ import annotations


def build_charge_of() -> tuple[list[int], int]:
    """Build the flat per-index charge list and the 0-based MASK index.

    Returns
    -------
    charge_of
        Integer oxidation state per 0-based vocab index, length
        ``num_species + 1``.  The final entry (MASK) is ``0``.
    mask_idx
        0-based index of the MASK token, i.e. ``num_species``.
    """
    from neutral_layer.data.vocab import build_species_vocab

    vocab = build_species_vocab()

    # 0-based index k corresponds to 1-based species index k+1.
    charge_of = [vocab.oxidation_state_of(i + 1) for i in range(vocab.num_species)]

    # MASK token: last column, charge 0.
    charge_of.append(0)
    mask_idx = vocab.num_species

    return charge_of, mask_idx
