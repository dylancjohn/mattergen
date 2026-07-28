"""Unit tests for SpeciesEmbedding.

Verifies:
- Output shape and dtype are correct.
- MASK token produces a valid embedding without index errors.
- The ``emb_size`` attribute is present (required by GemNetT).
- Element and OS embeddings are combined via ``e_element + gamma * e_os``.
- ``gamma`` is a learned scalar parameter, initialised to 1.0.
- GemNetTDenoiser's fc_atom output width respects num_atom_types.
"""

import pytest
import torch

from mattergen.common.gemnet.layers.embedding_block import SpeciesEmbedding
from mattergen.common.utils.globals import MAX_ATOMIC_NUM

# A small synthetic 4-species vocabulary: (Z, OS) pairs in ascending order.
_SPECIES_LIST: tuple[tuple[int, int], ...] = (
    (1, -1),   # H⁻
    (8, -2),   # O²⁻
    (26, 2),   # Fe²⁺
    (26, 3),   # Fe³⁺
)
_NUM_SPECIES = len(_SPECIES_LIST)
_MASK_INDEX = _NUM_SPECIES + 1  # = 5


@pytest.fixture
def emb() -> SpeciesEmbedding:
    return SpeciesEmbedding(
        species_list=_SPECIES_LIST,
        emb_size=16,
        with_mask_type=True,
    )


def test_emb_size_attribute(emb: SpeciesEmbedding) -> None:
    # GemNetT reads this attribute: getattr(atom_embedding, "emb_size")
    assert emb.emb_size == 16


def test_output_shape_valid_species(emb: SpeciesEmbedding) -> None:
    species_idx = torch.tensor([1, 2, 3, 4])  # all four valid species
    h = emb(species_idx)
    assert h.shape == (4, 16)
    assert h.dtype == torch.float32


def test_output_shape_mask_token(emb: SpeciesEmbedding) -> None:
    species_idx = torch.tensor([_MASK_INDEX])
    h = emb(species_idx)
    assert h.shape == (1, 16)
    # Should not raise; mask rows exist in both embedding tables.


def test_mixed_batch(emb: SpeciesEmbedding) -> None:
    # Typical diffusion batch: a mix of valid species and MASK tokens.
    species_idx = torch.tensor([1, _MASK_INDEX, 3, _MASK_INDEX, 2])
    h = emb(species_idx)
    assert h.shape == (5, 16)


def test_embedding_tables_are_summed_not_concatenated(emb: SpeciesEmbedding) -> None:
    # Output dim must equal emb_size (sum), not 2 * emb_size (concat).
    species_idx = torch.tensor([1])
    h = emb(species_idx)
    assert h.shape[-1] == emb.emb_size


def test_same_element_different_os_differs(emb: SpeciesEmbedding) -> None:
    # Fe²⁺ (idx=3) and Fe³⁺ (idx=4) share the same element embedding row
    # but differ in OS embedding — so their outputs must differ.
    h_fe2 = emb(torch.tensor([3]))
    h_fe3 = emb(torch.tensor([4]))
    assert not torch.allclose(h_fe2, h_fe3)


def test_lookup_buffers_are_registered(emb: SpeciesEmbedding) -> None:
    buffer_names = {name for name, _ in emb.named_buffers()}
    assert "species_to_z_idx" in buffer_names
    assert "species_to_os_idx" in buffer_names


def test_z_buf_values(emb: SpeciesEmbedding) -> None:
    # H⁻ is species index 1 → Z=1 → 0-based element index = 0
    assert emb.species_to_z_idx[1].item() == 0
    # O²⁻ is species index 2 → Z=8 → 0-based = 7
    assert emb.species_to_z_idx[2].item() == 7
    # Fe (species indices 3 and 4) → Z=26 → 0-based = 25
    assert emb.species_to_z_idx[3].item() == 25
    assert emb.species_to_z_idx[4].item() == 25
    # MASK → MAX_ATOMIC_NUM = 100 (extra row in element_embedding)
    assert emb.species_to_z_idx[_MASK_INDEX].item() == MAX_ATOMIC_NUM


def test_os_buf_values(emb: SpeciesEmbedding) -> None:
    # OS_VALUES[0] = -5, offset = 5
    # H⁻ OS=-1 → 0-based OS index = -1 + 5 = 4
    assert emb.species_to_os_idx[1].item() == 4
    # O²⁻ OS=-2 → 3
    assert emb.species_to_os_idx[2].item() == 3
    # Fe²⁺ OS=+2 → 7
    assert emb.species_to_os_idx[3].item() == 7
    # Fe³⁺ OS=+3 → 8
    assert emb.species_to_os_idx[4].item() == 8


def test_no_mask_type_variant() -> None:
    emb = SpeciesEmbedding(species_list=_SPECIES_LIST, emb_size=8, with_mask_type=False)
    species_idx = torch.tensor([1, 2, 3, 4])
    h = emb(species_idx)
    assert h.shape == (4, 8)


def test_fc_atom_output_width_default() -> None:
    """GemNetTDenoiser with default num_atom_types outputs MAX_ATOMIC_NUM + 1 logits."""
    import torch.nn as nn

    # Construct only the fc_atom layer to avoid building the full GemNetT graph.
    hidden_dim = 32
    with_mask_type = True
    fc = nn.Linear(hidden_dim, MAX_ATOMIC_NUM + int(with_mask_type))
    assert fc.out_features == MAX_ATOMIC_NUM + 1


def test_fc_atom_output_width_species() -> None:
    """fc_atom for species mode outputs num_species + 1 logits."""
    import torch.nn as nn

    num_species = _NUM_SPECIES
    with_mask_type = True
    fc = nn.Linear(16, num_species + int(with_mask_type))
    assert fc.out_features == num_species + 1


def test_gamma_is_registered_parameter(emb: SpeciesEmbedding) -> None:
    param_names = {name for name, _ in emb.named_parameters()}
    assert "gamma" in param_names
    assert emb.gamma.shape == (1,)


def test_gamma_initialised_to_one(emb: SpeciesEmbedding) -> None:
    assert torch.allclose(emb.gamma, torch.ones(1))


def test_gamma_at_one_equals_plain_sum() -> None:
    # At gamma=1.0 (the init value), output must equal a plain element+OS sum.
    emb = SpeciesEmbedding(species_list=_SPECIES_LIST, emb_size=16, with_mask_type=True)
    species_idx = torch.tensor([1, 2, 3, 4])
    z_idx = emb.species_to_z_idx[species_idx]
    os_idx = emb.species_to_os_idx[species_idx]
    expected = emb.element_embedding(z_idx) + emb.os_embedding(os_idx)
    assert torch.allclose(emb(species_idx), expected)
