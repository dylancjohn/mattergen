"""Crystal dataset over the species vocabulary.

Each atom type is a species, an (element, oxidation state) pair. Species
indices are 1-based and are stored in ``ChemGraph.atomic_numbers`` in place of
atomic numbers.
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

# Bounds come from neutral_layer.data.vocab.OS_VALUES, the single source of truth.
_OS_MIN: int = min(OS_VALUES)
_OS_MAX: int = max(OS_VALUES)
_OS_OFFSET: int = -_OS_MIN  # shift so _OS_MIN maps to column 0
_OS_COLS: int = _OS_MAX - _OS_MIN + 1  # 14 columns

# Upper bound on atomic numbers, matching MAX_ATOMIC_NUM in mattergen.common.utils.globals.
_Z_MAX: int = MAX_ATOMIC_NUM


def _build_species_indices(
    atomic_numbers: numpy.typing.NDArray,
    oxidation_states: numpy.typing.NDArray,
    vocab: SpeciesVocab,
) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
    """Map flat ``[N_atoms]`` atomic numbers and oxidation states to species indices.

    Returns ``(species_indices, invalid_mask)``, both ``[N_atoms]``. Indices are
    1-based; ``invalid_mask`` is True, and the index 0, where the (z, os) pair
    is not in the vocabulary. Callers decide whether to raise or filter.
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
    """Species-vocabulary counterpart of ``CrystalDataset``.

    Stores 1-based ``species_indices`` ``[N_atoms_total]`` and returns them as
    ``ChemGraph.atomic_numbers``. ``pos`` is ``[N_atoms_total, 3]`` fractional
    coordinates; ``cell``, ``num_atoms``, ``structure_id`` and each property
    array are per structure. Build with ``from_cache_path``.
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
        """Load from the standard MatterGen ``.npy`` files plus ``oxidation_states.npy``.

        Properties are read from ``{name}.json`` in ``cache_path``; a missing
        file raises ``FileNotFoundError``. The data is expected to be
        preprocessed already, so any structure that breaks one of the following
        rules raises ``DatasetValidationError`` rather than being dropped:

        1. more than ``max_atoms`` atoms (``MAX_ATOMS``);
        2. metal-only with a non-zero OS (``ALLOY_NONZERO_OS``);
        3. a single non-metal element (``SINGLE_SPECIES``);
        4. a (z, os) pair not in ``vocab`` (``UNKNOWN_SPECIES``);
        5. non-zero total charge (``CHARGE_NON_NEUTRAL``);
        6. an element outside ``mv_elements`` with more than one distinct OS
           (``MIXED_VALENCE``); skipped when ``mv_elements`` is None.
        """
        cache_path = str(cache_path)

        def _load(filename: str) -> numpy.typing.NDArray:
            path = os.path.join(cache_path, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
            # structure_id is stored as a numpy object array of strings.
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
        # Pure metals (e.g. Fe, Cu) are valid with OS=0, which Rule 2 has
        # already enforced. A lone non-metal has no counter-ion, so its OS is
        # undefined.
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
        # Once its atoms are committed, a non-neutral structure has log Z = -inf
        # under the structured output layer, giving NaN gradients in the
        # structured loss.
        charge_per_struct = np.zeros(len(num_atoms), dtype=np.int64)
        np.add.at(charge_per_struct, atom_struct_idx, oxidation_states)
        _raise_if_any(
            charge_per_struct != 0,
            ViolationCode.CHARGE_NON_NEUTRAL,
            "are non-neutral (total OS != 0)",
        )

        # ── Rule 6: MV element registry ──────────────────────────────────────
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
