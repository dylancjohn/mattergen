# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import time

import numpy as np
from ase import Atoms
from ase.io import write
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from mattersim.utils.logger_utils import get_logger
from pymatgen.core import Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from mattergen.common.utils.globals import get_device

logger = get_logger()
logger.level("ERROR")


# mattersim's BatchRelaxer only exits once every structure has converged, so a
# few that never do can hold a job until walltime with nothing written. Two
# optional caps, both off by default because they change the reported metrics:
#
#   MATTERGEN_RELAX_MAX_STEPS    per-structure optimizer-step cap. Steps vary
#       widely (a 500-step cap retires 18-25% of some baselines' samples), so
#       prefer the time budget.
#   MATTERGEN_RELAX_MAX_SECONDS  wall-clock budget for the relaxation phase,
#       after which unfinished structures are retired. Leave ~1 h per 10k
#       structures of walltime for the metrics phase.
#
# Retired structures count as failed jobs (see relax_atoms).
def _env_number(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Ignoring non-numeric {name}={raw!r}; using {default}.")
        return default


class CappedBatchRelaxer(BatchRelaxer):
    """BatchRelaxer with optional step and wall-clock caps.

    `capped_indices` holds the indices (into the list passed to `relax`) of
    retired structures; ones never started are absent from the trajectories.
    """

    def __init__(self, *args, max_steps: int | None = None, max_seconds: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        self.max_seconds = max_seconds if max_seconds and max_seconds > 0 else None
        self.capped_indices: set[int] = set()
        self.n_never_started = 0
        self._steps: dict[int, int] = {}
        self._deadline: float | None = None

    def _out_of_time(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def _apply_step_cap(self) -> None:
        """Retire instances that have used their step budget.

        Runs after step_batch(), so every remaining instance has just stepped
        and is unconverged.
        """
        if self.max_steps is None:
            return
        survivors = []
        for opt in self.optimizer_instances:
            index = opt.atoms.info["structure_index"]
            self._steps[index] = self._steps.get(index, 0) + 1
            if self._steps[index] >= self.max_steps:
                self.capped_indices.add(index)
            else:
                survivors.append(opt)
        if len(survivors) != len(self.optimizer_instances):
            self.optimizer_instances = survivors
            self.is_active_instance = [True] * len(survivors)
            # step_batch() left finished=False for these; avoid a deadlock.
            if not self.optimizer_instances:
                self.finished = True

    def relax(self, atoms_list: list[Atoms]) -> dict[int, list[Atoms]]:
        # BatchRelaxer.relax with the caps added. Reimplemented rather than
        # wrapped because the deadline must also stop new insertions.
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        if self.max_seconds is not None:
            self._deadline = time.monotonic() + self.max_seconds
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while pointer < len(atoms_list) or not self.finished:
            if self._out_of_time():
                for opt in self.optimizer_instances:
                    self.capped_indices.add(opt.atoms.info["structure_index"])
                self.n_never_started = len(atoms_list) - pointer
                logger.warning(
                    f"Relaxation wall-clock budget of {self.max_seconds:.0f} s expired: "
                    f"retiring {len(self.optimizer_instances)} structure(s) still relaxing "
                    f"and {self.n_never_started} not yet started."
                )
                self.optimizer_instances = []
                self.is_active_instance = []
                self.finished = True
                break
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                self.insert(atoms_list_[pointer])
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
            self._apply_step_cap()
        self.tqdmcounter.close()

        return self.trajectories


def relax_atoms(
    atoms: list[Atoms], device: str = str(get_device()), potential_load_path: str = None, output_path: str | None = None, **kwargs
) -> tuple[list[Atoms], np.ndarray, list[int]]:
    # Skip degenerate cells, which crash every structure batched with them: a
    # collapsed cell exhausts GPU memory in the fixed-cutoff neighbour graph, and
    # an exploded one in pymatgen's neighbour-list binning, which scales with
    # absolute volume. The per-atom bounds are far outside real crystals
    # (diamond is 5.7 A^3/atom; MP-20-OS peaks at 134). Skipped and retired
    # structures are left out of keep_idx, which evaluate.py turns into
    # n_failed_jobs, so they stay in the metric denominators.
    MIN_VOLUME_PER_ATOM_A3 = 2.0
    MAX_VOLUME_PER_ATOM_A3 = 1000.0
    keep_idx: list[int] = []
    n_empty = n_collapsed = n_exploded = 0
    for i, a in enumerate(atoms):
        if len(a) == 0:
            n_empty += 1
            continue
        volume_per_atom = a.get_volume() / len(a)
        if volume_per_atom < MIN_VOLUME_PER_ATOM_A3:
            n_collapsed += 1
        elif volume_per_atom > MAX_VOLUME_PER_ATOM_A3:
            n_exploded += 1
        else:
            keep_idx.append(i)
    if len(keep_idx) < len(atoms):
        logger.warning(
            f"Skipping relaxation for {len(atoms) - len(keep_idx)} of {len(atoms)} "
            f"structure(s) with a degenerate unit cell "
            f"({n_collapsed} below {MIN_VOLUME_PER_ATOM_A3} A^3/atom, "
            f"{n_exploded} above {MAX_VOLUME_PER_ATOM_A3} A^3/atom, "
            f"{n_empty} with no atoms). "
            f"These are counted as failed jobs, not removed from the denominator."
        )

    potential = Potential.from_checkpoint(
        device=device, load_path=potential_load_path, load_training_state=False
    )
    max_steps = int(_env_number("MATTERGEN_RELAX_MAX_STEPS", 0))
    max_seconds = _env_number("MATTERGEN_RELAX_MAX_SECONDS", 0)
    logger.warning(
        f"Relaxing {len(keep_idx)} structure(s) with "
        f"max_steps={max_steps or 'off'}, max_seconds={max_seconds or 'off'}."
    )
    batch_relaxer = CappedBatchRelaxer(
        potential=potential,
        filter="EXPCELLFILTER",
        max_steps=max_steps,
        max_seconds=max_seconds,
        **kwargs,
    )
    relaxation_trajectories = batch_relaxer.relax([atoms[i] for i in keep_idx])

    # Keep only converged structures, in input order.
    converged_local_idx = sorted(
        i for i in relaxation_trajectories if i not in batch_relaxer.capped_indices
    )
    n_retired = len(keep_idx) - len(converged_local_idx)
    if n_retired:
        n_stepped_out = len(batch_relaxer.capped_indices)
        logger.warning(
            f"Retiring {n_retired} of {len(keep_idx)} structure(s) that did not reach "
            f"fmax<{batch_relaxer.fmax} "
            f"({n_stepped_out} hit a cap, {batch_relaxer.n_never_started} never started). "
            f"These are counted as failed jobs, not removed from the denominator. "
            f"An uncapped run would differ only in these {n_retired} "
            f"({100.0 * n_retired / max(len(keep_idx), 1):.2f}% of relaxed structures)."
        )

    relaxed_atoms = [relaxation_trajectories[i][-1] for i in converged_local_idx]
    total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])
    keep_idx = [keep_idx[i] for i in converged_local_idx]
    if output_path:
        write(output_path, relaxed_atoms, format="extxyz")
        logger.info(f"Relaxed structures saved to {output_path}")
    return relaxed_atoms, total_energies, keep_idx


def relax_structures(
    structures: Structure | list[Structure],
    device: str = str(get_device()),
    potential_load_path: str = None,
    output_path: str | None = None,
    **kwargs,
) -> tuple[list[Structure], np.ndarray, list[int]]:
    """Relax structures with the MatterSim force field.

    Returns `(relaxed_structures, energies, keep_idx)`, where `keep_idx` indexes
    the input structures that were relaxed (degenerate and capped ones are
    dropped); subset any other per-structure list by it. Oxidation states are
    re-attached to the returned structures but not written to the extxyz file.
    """
    if isinstance(structures, Structure):
        structures = [structures]

    # None for structures with plain Element (not Species) sites.
    saved_os: list[list[float] | None] = []
    for s in structures:
        if any(isinstance(site.specie, Species) for site in s):
            saved_os.append([site.specie.oxi_state for site in s])
        else:
            saved_os.append(None)

    atoms = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    relaxed_atoms, total_energies, keep_idx = relax_atoms(
        atoms, device=device, potential_load_path=potential_load_path, output_path=output_path, **kwargs
    )
    relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]

    # relax_atoms may have skipped structures, so align saved_os via keep_idx.
    for structure, os_list in zip(relaxed_structures, (saved_os[i] for i in keep_idx)):
        if os_list is not None:
            structure.add_oxidation_state_by_site(os_list)

    return relaxed_structures, total_energies, keep_idx
