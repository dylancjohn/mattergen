"""test_dataset.py

Unit tests for mattergen/constraints/dataset.py.

Tests cover:
    - Loading a dataset from fake .npy files via from_cache_path.
    - __getitem__ returns species indices in the correct range.
    - Atom count consistency between atomic_numbers shape and num_atoms.
    - Species correctness: decoded (z, os) matches original npy values.
    - subset produces a correctly sized dataset with matching indices.
    - ValueError raised for unknown (atomic_number, os) pairs.
    - FileNotFoundError raised for missing .npy files.
    - Filter: max_atoms drops oversized structures.
    - Filter: alloy check keeps metallic-only OS=0 structures, drops non-zero OS.
    - Filter: single-species drops elemental compounds.
    - Filter: charge neutrality drops non-neutral structures.
    - Filter: MV element registry drops non-MV elements with >1 distinct OS.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from mattergen.constraints.dataset import SpeciesCrystalDataset
from neutral_layer.vocab import SpeciesVocab

# Synthetic vocab: (8,-2)->1, (26,0)->2, (26,2)->3, (26,3)->4, (28,0)->5, mask->6
# Includes Fe and Ni at OS=0 to enable two-element metallic alloy tests.
_VOCAB = SpeciesVocab(
    species_list=((8, -2), (26, 0), (26, 2), (26, 3), (28, 0))
)


def _write_npy_files(
    directory: Path,
    atomic_numbers: list[int],
    oxidation_states: list[int],
    num_atoms: list[int],
) -> None:
    """Write minimal .npy files for a fake crystal dataset."""
    n_structures = len(num_atoms)
    n_atoms = sum(num_atoms)

    np.save(directory / "atomic_numbers.npy", np.array(atomic_numbers, dtype=np.int64))
    np.save(directory / "oxidation_states.npy", np.array(oxidation_states, dtype=np.int64))
    np.save(directory / "num_atoms.npy", np.array(num_atoms, dtype=np.int64))
    np.save(directory / "pos.npy", np.zeros((n_atoms, 3), dtype=np.float32))
    np.save(
        directory / "cell.npy",
        np.tile(np.eye(3, dtype=np.float32), (n_structures, 1, 1)).reshape(n_structures, 3, 3),
    )
    np.save(directory / "structure_id.npy", np.arange(n_structures, dtype=np.int64))


@pytest.fixture
def fake_dataset_path(tmp_path: Path) -> Path:
    """Two charge-neutral ionic structures.

    Structure 0: (8,-2), (26,2)   charge: -2+2=0   species indices [1, 3]
    Structure 1: (8,-2), (26,2)   charge: -2+2=0   species indices [1, 3]
    """
    _write_npy_files(
        tmp_path,
        atomic_numbers=[8, 26, 8, 26],
        oxidation_states=[-2, 2, -2, 2],
        num_atoms=[2, 2],
    )
    return tmp_path


@pytest.fixture
def dataset(fake_dataset_path: Path) -> SpeciesCrystalDataset:
    return SpeciesCrystalDataset.from_cache_path(fake_dataset_path, _VOCAB)


class TestFromCachePath:
    def test_length(self, dataset: SpeciesCrystalDataset):
        assert len(dataset) == 2

    def test_missing_file_raises(self, tmp_path: Path):
        _write_npy_files(tmp_path, [8], [-2], [1])
        (tmp_path / "oxidation_states.npy").unlink()
        with pytest.raises(FileNotFoundError, match="oxidation_states.npy"):
            SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)

    def test_unknown_pair_raises_when_strict(self, tmp_path: Path):
        _write_npy_files(tmp_path, [8, 26], [-2, 99], [2])
        with pytest.raises(ValueError, match="not in the species vocabulary"):
            SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB, filter_unknown_species=False)

    def test_unknown_pair_filtered_by_default(self, tmp_path: Path):
        # One valid structure (O²⁻ + Fe²⁺) and one invalid (O²⁻ + Z=26 OS=99).
        # The invalid structure is silently dropped; the valid one is kept.
        _write_npy_files(tmp_path, [8, 26, 8, 26], [-2, 2, -2, 99], [2, 2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 1, f"Expected 1 structure after filtering, got {len(ds)}"


class TestGetItem:
    def test_indices_in_range(self, dataset: SpeciesCrystalDataset):
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample.atomic_numbers.min() >= 1
            assert sample.atomic_numbers.max() <= _VOCAB.num_species

    def test_atom_count_consistency(self, dataset: SpeciesCrystalDataset):
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample.atomic_numbers.shape[0] == int(sample.num_atoms)

    def test_species_indices_structure_0(self, dataset: SpeciesCrystalDataset):
        # (8,-2)->1, (26,2)->3
        assert dataset[0].atomic_numbers.tolist() == [1, 3]

    def test_species_indices_structure_1(self, dataset: SpeciesCrystalDataset):
        # (8,-2)->1, (26,2)->3
        assert dataset[1].atomic_numbers.tolist() == [1, 3]

    def test_atomic_numbers_dtype(self, dataset: SpeciesCrystalDataset):
        sample = dataset[0]
        assert sample.atomic_numbers.dtype == torch.int64

    def test_pos_shape(self, dataset: SpeciesCrystalDataset):
        sample = dataset[0]
        assert sample.pos.shape == (2, 3)

    def test_cell_shape(self, dataset: SpeciesCrystalDataset):
        sample = dataset[0]
        assert sample.cell.shape == (1, 3, 3)


class TestSubset:
    def test_subset_length(self, dataset: SpeciesCrystalDataset):
        sub = dataset.subset([0])
        assert len(sub) == 1

    def test_subset_species_match(self, dataset: SpeciesCrystalDataset):
        sub = dataset.subset([0])
        assert sub[0].atomic_numbers.tolist() == dataset[0].atomic_numbers.tolist()

    def test_subset_both_structures(self, dataset: SpeciesCrystalDataset):
        sub = dataset.subset([0, 1])
        assert len(sub) == 2
        assert sub[0].atomic_numbers.tolist() == dataset[0].atomic_numbers.tolist()
        assert sub[1].atomic_numbers.tolist() == dataset[1].atomic_numbers.tolist()

    def test_subset_vocab_preserved(self, dataset: SpeciesCrystalDataset):
        sub = dataset.subset([1])
        assert sub.vocab is _VOCAB


class TestSpeciesDecodingRoundTrip:
    def test_decode_all_atoms(self, fake_dataset_path: Path):
        """Decoded (z, os) pairs must match the original atomic_numbers.npy and oxidation_states.npy."""
        dataset = SpeciesCrystalDataset.from_cache_path(fake_dataset_path, _VOCAB)
        raw_z = np.load(fake_dataset_path / "atomic_numbers.npy")
        raw_os = np.load(fake_dataset_path / "oxidation_states.npy")

        atom_idx = 0
        for struct_idx in range(len(dataset)):
            sample = dataset[struct_idx]
            n = int(sample.num_atoms)
            for local_idx in range(n):
                species_idx = sample.atomic_numbers[local_idx].item()
                assert _VOCAB.atomic_number_of(species_idx) == raw_z[atom_idx]
                assert _VOCAB.oxidation_state_of(species_idx) == raw_os[atom_idx]
                atom_idx += 1


class TestNeutralityFilter:
    def test_non_neutral_structure_dropped(self, tmp_path: Path):
        """Structures with non-zero total OS must be silently dropped."""
        # Structure 0: (8,-2) + (26,2) = 0  (neutral, kept)
        # Structure 1: (8,-2) + (26,3) = +1 (non-neutral, dropped)
        _write_npy_files(tmp_path, [8, 26, 8, 26], [-2, 2, -2, 3], [2, 2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 1, f"Expected 1 structure after neutrality filter, got {len(ds)}"
        assert ds[0].atomic_numbers.tolist() == [1, 3]

    def test_all_neutral_kept(self, tmp_path: Path):
        """When all structures are neutral, none should be dropped."""
        _write_npy_files(tmp_path, [8, 26, 8, 26], [-2, 2, -2, 2], [2, 2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 2

    def test_all_non_neutral_dropped(self, tmp_path: Path):
        """When every structure is non-neutral, the dataset is empty."""
        _write_npy_files(tmp_path, [8, 26], [-2, 3], [2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 0


class TestMaxAtomsFilter:
    def test_oversized_structure_dropped(self, tmp_path: Path):
        """Structures with more atoms than max_atoms must be dropped."""
        # 10-atom structure, threshold = 4 → dropped
        _write_npy_files(tmp_path, [8, 26] * 5, [-2, 2] * 5, [10])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB, max_atoms=4)
        assert len(ds) == 0

    def test_at_limit_kept(self, tmp_path: Path):
        """Structures exactly at max_atoms must be kept."""
        _write_npy_files(tmp_path, [8, 26, 8, 26], [-2, 2, -2, 2], [4])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB, max_atoms=4)
        assert len(ds) == 1

    def test_mixed_sizes(self, tmp_path: Path):
        """Only structures within the limit should survive."""
        # Structure 0: 2 atoms (kept), Structure 1: 6 atoms (dropped at max_atoms=4)
        _write_npy_files(tmp_path, [8, 26] + [8, 26] * 3, [-2, 2] + [-2, 2] * 3, [2, 6])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB, max_atoms=4)
        assert len(ds) == 1


class TestAlloyFilter:
    def test_metallic_all_zero_os_kept(self, tmp_path: Path):
        """Two-element metallic alloy with all OS=0 must be kept."""
        # Fe + Ni, both OS=0: passes alloy check and single-species filter.
        _write_npy_files(tmp_path, [26, 28], [0, 0], [2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 1

    def test_metallic_nonzero_os_dropped(self, tmp_path: Path):
        """Pure-metal structure with non-zero OS must be dropped."""
        _write_npy_files(tmp_path, [26, 26], [2, -2], [2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 0

    def test_ionic_compound_not_affected(self, tmp_path: Path):
        """Non-metallic compound (contains O) must pass through the alloy filter."""
        _write_npy_files(tmp_path, [8, 26], [-2, 2], [2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 1


class TestSingleSpeciesFilter:
    def test_single_element_dropped(self, tmp_path: Path):
        """Structures containing only one distinct element must be dropped."""
        # Only O atoms (not an alloy since O is non-metal, but single-species)
        # Two O atoms with -2, total charge = -4 (also non-neutral, but filter fires first)
        # Use a neutral single-element case to isolate the filter: that's impossible for non-metals.
        # Use (26, 0) × 2: single metal species, total charge = 0.
        _write_npy_files(tmp_path, [26, 26], [0, 0], [2])
        # The alloy filter keeps this (all metal, all OS=0), but single-species drops it.
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 0

    def test_two_element_compound_kept(self, tmp_path: Path):
        """Structures with at least two distinct elements must survive."""
        _write_npy_files(tmp_path, [8, 26], [-2, 2], [2])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 1


class TestMVFilter:
    def test_non_mv_element_two_os_dropped(self, tmp_path: Path):
        """Structure where a non-MV element (O) carries two distinct OS values must be dropped."""
        # O appears as -2 and -1; O is not in MIXED_VALENCE_ELEMENTS
        # Charge: -2 + -1 + 3 = 0
        _write_npy_files(tmp_path, [8, 8, 26], [-2, -1, 3], [3])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 0

    def test_mv_element_two_os_kept(self, tmp_path: Path):
        """Structure where a MV element (Fe) carries two distinct OS values must be kept.

        Fe3O4: Fe(2+) Fe(3+) Fe(3+) O(2-)×4 → charge: 2+3+3-8=0 ✓
        Fe is in MIXED_VALENCE_ELEMENTS; O has only one OS value (-2).
        """
        _write_npy_files(
            tmp_path,
            [26, 26, 26, 8, 8, 8, 8],
            [2, 3, 3, -2, -2, -2, -2],
            [7],
        )
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, _VOCAB)
        assert len(ds) == 1

    def test_mv_filter_disabled(self, tmp_path: Path):
        """With mv_elements=None the MV filter is skipped and the structure is kept."""
        # O appears with -2 and -1 (non-MV violation). Use a local vocab that includes
        # (8, -1) so the unknown-species filter doesn't intercept it first.
        local_vocab = SpeciesVocab(species_list=((8, -2), (8, -1), (26, 3)))
        # Charge: -2 + -1 + 3 = 0 ✓
        _write_npy_files(tmp_path, [8, 8, 26], [-2, -1, 3], [3])
        ds = SpeciesCrystalDataset.from_cache_path(tmp_path, local_vocab, mv_elements=None)
        assert len(ds) == 1
