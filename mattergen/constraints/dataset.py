"""dataset.py

SpeciesCrystalDataset for training MatterGen on a species (element + OS) vocabulary.

Mirrors CrystalDataset but stores 1-based species indices in ChemGraph.atomic_numbers
rather than raw atomic numbers. Preserving the 1-based convention means AtomEmbedding
(which applies Z - 1) and D3PMCorruption (offset=1) are unchanged; only the vocab
size in model config needs to differ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing
import torch
from neutral_layer.data.classifiers import METAL_ATOMIC_NUMBERS
from neutral_layer.data.filtering import DatasetValidationError, ViolationCode
from neutral_layer.data.vocab import MIXED_VALENCE_ELEMENTS, OS_VALUES, SpeciesVocab

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.dataset import CORE_STRUCTURE_FILE_NAMES, BaseDataset
from mattergen.common.data.transform import Transform
from mattergen.common.data.types import PropertySourceId, PropertyValues
from mattergen.common.utils.globals import MAX_ATOMIC_NUM, PROPERTY_SOURCE_IDS

OXIDATION_STATES_FILE = "oxidation_states.npy"

# Bounds matching neutral_layer.data.vocab.OS_VALUES (single source of truth).
_OS_MIN: int = min(OS_VALUES)
_OS_MAX: int = max(OS_VALUES)
_OS_OFFSET: int = -_OS_MIN  # shift so _OS_MIN maps to column 0
_OS_COLS: int = _OS_MAX - _OS_MIN + 1  # 14 columns

# Upper bound on atomic numbers; matches MAX_ATOMIC_NUM in mattergen.common.utils.globals.
_Z_MAX: int = MAX_ATOMIC_NUM


def _build_species_indices(
    atomic_numbers: numpy.typing.NDArray,
    oxidation_states: numpy.typing.NDArray,
    vocab: SpeciesVocab,
) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
    """Map parallel flat arrays of atomic numbers and OS to 1-based species indices.

    Uses a 2D lookup table for O(N) vectorised mapping. A value of 0 in the
    returned indices array marks atoms whose (z, os) pair is absent from the
    vocabulary; callers decide how to handle these (raise or filter).

    Parameters
    ----------
    atomic_numbers
        Flat int64 array of 1-based atomic numbers, shape [N_atoms].
    oxidation_states
        Flat int64 array of oxidation states, shape [N_atoms].
    vocab
        Species vocabulary defining valid (z, os) pairs and their indices.

    Returns
    -------
    species_indices
        Flat int64 array of 1-based species indices, shape [N_atoms].
        Entries are 0 for atoms with unknown (z, os) pairs.
    invalid_mask
        Boolean array, shape [N_atoms]. True where the (z, os) pair is not in vocab.
    """
    lookup = np.zeros((_Z_MAX + 1, _OS_COLS), dtype=np.int64)
    for (z, oxs), idx in vocab.species_to_idx.items():
        if 0 <= z <= _Z_MAX and _OS_MIN <= oxs <= _OS_MAX:
            lookup[z, oxs + _OS_OFFSET] = idx

    os_in_range = (oxidation_states >= _OS_MIN) & (oxidation_states <= _OS_MAX)
    z_in_range = (atomic_numbers >= 0) & (atomic_numbers <= _Z_MAX)

    safe_z = np.clip(atomic_numbers, 0, _Z_MAX)
    safe_os_col = np.clip(oxidation_states + _OS_OFFSET, 0, _OS_COLS - 1)
    indices = lookup[safe_z, safe_os_col]

    invalid_mask = (indices == 0) | ~os_in_range | ~z_in_range
    return indices, invalid_mask


@dataclass(frozen=True, kw_only=True)
class SpeciesCrystalDataset(BaseDataset):
    """Dataset for crystal structures using a species (element + OS) vocabulary.

    Mirrors CrystalDataset but stores 1-based species indices in ChemGraph.atomic_numbers.
    Use from_cache_path to load from a directory containing the standard MatterGen
    .npy files plus oxidation_states.npy.

    Attributes
    ----------
    pos
        Fractional coordinates, shape [N_atoms_total, 3].
    cell
        Lattice matrices, shape [N_structures, 3, 3].
    species_indices
        1-based species indices, shape [N_atoms_total]. Precomputed at load time.
    num_atoms
        Number of atoms per structure, shape [N_structures].
    structure_id
        Structure identifiers, shape [N_structures].
    vocab
        Species vocabulary used to interpret species_indices.
    properties
        Per-structure property arrays, keyed by PropertySourceId.
    transforms
        Per-sample transforms applied sequentially in __getitem__.
    """

    pos: numpy.typing.NDArray
    cell: numpy.typing.NDArray
    species_indices: numpy.typing.NDArray
    num_atoms: numpy.typing.NDArray
    structure_id: numpy.typing.NDArray
    vocab: SpeciesVocab
    properties: dict[PropertySourceId, numpy.typing.NDArray] = field(
        default_factory=dict
    )
    transforms: list[Transform] | None = None

    def __post_init__(self) -> None:
        property_names = list(self.properties.keys())
        assert all(s in PROPERTY_SOURCE_IDS for s in property_names), (
            f"Property names {property_names} are not valid. "
            f"Valid property source names: {PROPERTY_SOURCE_IDS}"
        )

    @cached_property
    def index_offset(self) -> numpy.typing.NDArray:
        """Cumulative atom offset for indexing into pos and species_indices."""
        return np.concatenate([np.array([0]), np.cumsum(self.num_atoms[:-1])])

    def __getitem__(self, index: int) -> ChemGraph:
        pos_offset = self.index_offset[index]
        num_atoms = torch.tensor(self.num_atoms[index])

        props_dict = self.get_properties_dict(index)
        data = ChemGraph(
            pos=torch.from_numpy(self.pos[pos_offset : pos_offset + num_atoms]).float()
            % 1.0,
            cell=torch.from_numpy(self.cell[index]).float().unsqueeze(0),
            atomic_numbers=torch.from_numpy(
                self.species_indices[pos_offset : pos_offset + num_atoms]
            ),
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            **props_dict,  # type: ignore[arg-type]
        )

        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)
        return data

    def __len__(self) -> int:
        return len(self.num_atoms)

    def subset(self, indices: Sequence[int]) -> "SpeciesCrystalDataset":
        """Return a new dataset containing only the structures at the given indices."""
        batch_indices: list[int] = []
        for index in indices:
            pos_offset = self.index_offset[index]
            batch_indices.extend(range(pos_offset, pos_offset + self.num_atoms[index]))

        idx_list = list(indices)
        return SpeciesCrystalDataset(
            pos=self.pos[batch_indices],
            cell=self.cell[idx_list],
            species_indices=self.species_indices[batch_indices],
            num_atoms=self.num_atoms[idx_list],
            structure_id=self.structure_id[idx_list],
            vocab=self.vocab,
            properties={k: v[idx_list] for k, v in self.properties.items()},
            transforms=self.transforms,
        )

    @classmethod
    def from_cache_path(  # type: ignore[override]
        cls,
        cache_path: str | Path,
        vocab: SpeciesVocab,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
        max_atoms: int = 200,
        mv_elements: frozenset[str] | None = MIXED_VALENCE_ELEMENTS,
    ) -> "SpeciesCrystalDataset":
        """Load a SpeciesCrystalDataset from a directory of .npy files.

        Reads the standard MatterGen files (pos.npy, cell.npy, atomic_numbers.npy,
        num_atoms.npy, structure_id.npy) plus oxidation_states.npy, then maps each
        (atomic_number, os) pair to a 1-based species index via vocab. Six rules
        are validated; a structure violating any rule raises immediately rather than being
        silently dropped, since this data should already have been through preprocessing.

        Parameters
        ----------
        cache_path
            Directory containing the .npy files.
        vocab
            Species vocabulary for mapping (atomic_number, os) to indices.
        transforms
            Per-sample transforms applied in __getitem__.
        properties
            Property names to load from {prop}.json files in cache_path.
        max_atoms
            Structures with more atoms than this threshold raise
            ``DatasetValidationError`` (``ViolationCode.MAX_ATOMS``).
        mv_elements
            Elements permitted to carry more than one distinct OS per compound.
            Structures where any other element appears with multiple OS values
            raise ``DatasetValidationError`` (``ViolationCode.MIXED_VALENCE``).
            Pass ``None`` to disable this check.

        Returns
        -------
        dataset
            Loaded SpeciesCrystalDataset.

        Raises
        ------
        FileNotFoundError
            If any required .npy file or property .json file is missing.
        neutral_layer.data.filtering.DatasetValidationError
            If any structure violates one of the 6 rules above (see module
            docstring for the specific ViolationCode raised per rule).
        """
        cache_path = str(cache_path)

        def _load(filename: str) -> numpy.typing.NDArray:
            path = os.path.join(cache_path, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
            # allow_pickle=True is needed for structure_id arrays which are stored
            # as numpy object arrays (string IDs like "mp-1234").
            return np.load(path, allow_pickle=True)

        pos = _load(CORE_STRUCTURE_FILE_NAMES["pos"])
        cell = _load(CORE_STRUCTURE_FILE_NAMES["cell"])
        raw_atomic_numbers = _load(CORE_STRUCTURE_FILE_NAMES["atomic_numbers"])
        num_atoms = _load(CORE_STRUCTURE_FILE_NAMES["num_atoms"])
        structure_id = _load(CORE_STRUCTURE_FILE_NAMES["structure_id"])
        oxidation_states = _load(OXIDATION_STATES_FILE)

        species_indices, invalid_atom_mask = _build_species_indices(
            raw_atomic_numbers, oxidation_states, vocab
        )

        # Build per-atom → per-structure mapping and cumulative atom offsets.
        atom_struct_idx = np.repeat(np.arange(len(num_atoms)), num_atoms)
        offsets = np.concatenate([[0], np.cumsum(num_atoms)])

        def _raise_if_any(
            bad_mask: numpy.typing.NDArray,
            violation: ViolationCode,
            summary: str,
        ) -> None:
            if not bad_mask.any():
                return
            bad_ids = structure_id[bad_mask].tolist()
            sample = bad_ids[:10]
            more = f" (+{len(bad_ids) - 10} more)" if len(bad_ids) > 10 else ""
            raise DatasetValidationError(
                violation,
                f"{cache_path}: {int(bad_mask.sum())}/{len(num_atoms)} structures {summary}: "
                f"{sample}{more}",
            )

        # ── Rule 1: max atoms ────────────────────────────────────────────────
        _raise_if_any(
            num_atoms > max_atoms,
            ViolationCode.MAX_ATOMS,
            f"exceed max_atoms={max_atoms}",
        )

        # ── Rule 2: alloy check ──────────────────────────────────────────────
        # Metallic-only structures are only valid when every atom carries OS=0.
        # Non-zero OS on a metal-only compound is an ICSD artefact.
        is_metal_atom = np.array(
            [z in METAL_ATOMIC_NUMBERS for z in raw_atomic_numbers.tolist()]
        )
        alloy_bad = np.zeros(len(num_atoms), dtype=bool)
        for i in range(len(num_atoms)):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if is_metal_atom[s:e].all():
                alloy_bad[i] = not (oxidation_states[s:e] == 0).all()
        _raise_if_any(
            alloy_bad,
            ViolationCode.ALLOY_NONZERO_OS,
            "are metallic-only with non-zero OS",
        )

        # ── Rule 3: single-species ───────────────────────────────────────────
        # Single-species alloys (e.g. elemental Fe, Cu) are valid -- OS=0 is a
        # well-defined answer for a pure metal, and Rule 2 above already
        # guarantees any metal-only structure reaching this point has OS=0.
        # Single-species non-metals are still rejected: OS is undefined for a
        # lone non-metal element with no counter-ion.
        single_species_bad = np.zeros(len(num_atoms), dtype=bool)
        for i in range(len(num_atoms)):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if len(set(raw_atomic_numbers[s:e].tolist())) == 1:
                single_species_bad[i] = not is_metal_atom[s:e].all()
        _raise_if_any(
            single_species_bad,
            ViolationCode.SINGLE_SPECIES,
            "are single-species non-metals",
        )

        # ── Rule 4: unknown (z, os) pairs ────────────────────────────────────
        if invalid_atom_mask.any():
            unknown_pairs: set[tuple[int, int]] = set(
                zip(
                    raw_atomic_numbers[invalid_atom_mask].tolist(),
                    oxidation_states[invalid_atom_mask].tolist(),
                )
            )
            invalid_count_per_struct = np.zeros(len(num_atoms), dtype=np.int64)
            np.add.at(
                invalid_count_per_struct,
                atom_struct_idx,
                invalid_atom_mask.astype(np.int64),
            )
            bad_from_unknown = invalid_count_per_struct > 0
            raise DatasetValidationError(
                ViolationCode.UNKNOWN_SPECIES,
                f"{cache_path}: {int(bad_from_unknown.sum())}/{len(num_atoms)} structures "
                f"contain (atomic_number, oxidation_state) pairs not in the species "
                f"vocabulary: {sorted(unknown_pairs)}",
            )

        # ── Rule 5: non-neutral structures ───────────────────────────────────
        # Non-neutral structures cause log_z = -inf for all timesteps once
        # their atoms are committed, triggering NaN in the SPL backward pass.
        charge_per_struct = np.zeros(len(num_atoms), dtype=np.int64)
        np.add.at(charge_per_struct, atom_struct_idx, oxidation_states)
        _raise_if_any(
            charge_per_struct != 0,
            ViolationCode.CHARGE_NON_NEUTRAL,
            "are non-neutral (total OS != 0)",
        )

        # ── Rule 6: MV element registry ──────────────────────────────────────
        # Non-MV elements must carry exactly one distinct OS per structure.
        if mv_elements is not None:
            from pymatgen.core import Element as _PmgEl

            z_to_sym: dict[int, str] = {e.Z: e.symbol for e in _PmgEl}
            mv_bad = np.zeros(len(num_atoms), dtype=bool)
            for i in range(len(num_atoms)):
                s, e = int(offsets[i]), int(offsets[i + 1])
                el_os: dict[str, set[int]] = {}
                for z, oxs in zip(
                    raw_atomic_numbers[s:e].tolist(), oxidation_states[s:e].tolist()
                ):
                    sym = z_to_sym.get(z, "")
                    el_os.setdefault(sym, set()).add(oxs)
                for sym, charges in el_os.items():
                    if len(charges) > 1 and sym not in mv_elements:
                        mv_bad[i] = True
                        break
            _raise_if_any(
                mv_bad,
                ViolationCode.MIXED_VALENCE,
                "have a non-MV element carrying >1 distinct OS",
            )

        property_names = properties or []
        props: dict[PropertySourceId, numpy.typing.NDArray] = {}
        for prop_name in property_names:
            prop_path = os.path.join(cache_path, f"{prop_name}.json")
            if not os.path.exists(prop_path):
                raise FileNotFoundError(
                    f"{prop_name}.json does not exist in {cache_path}."
                )
            props[prop_name] = PropertyValues.from_json(prop_path).values

        return cls(
            pos=pos,
            cell=cell,
            species_indices=species_indices,
            num_atoms=num_atoms,
            structure_id=structure_id,
            vocab=vocab,
            properties=props,
            transforms=transforms,
        )
