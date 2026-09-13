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


# Upstream BatchRelaxer has NO iteration cap. Its only exit is opt.converged()
# (fmax < 0.05) for *every* live instance, so a handful of structures that never
# converge keep the whole loop alive indefinitely: the job runs until the Slurm
# walltime kills it and writes nothing at all -- no relaxed structures, no
# metrics. This is not hypothetical; it cost two OMatG seeds an 8 h job each,
# both of which had relaxed >10200 of 10240 structures before stalling on the
# remainder for over four hours.
#
# It is also invisible in the log. BatchRelaxer's tqdm counter advances on
# *insertion into the batch*, not on convergence, so it reads 100% while all the
# convergence work is still outstanding.
#
# Defaults, both overridable by environment variable:
#
#   MATTERGEN_RELAX_MAX_STEPS    (default 500)
#       Per-structure cap on optimizer steps. 500 is not arbitrary: it is
#       mattersim's own default for its single-structure Relaxer
#       (mattersim/applications/relax.py), which likewise treats
#       `get_number_of_steps() >= steps` as "did not converge". So this makes
#       the batch relaxer agree with the non-batch one rather than inventing a
#       new convention. 0 disables the cap (upstream behaviour).
#
#   MATTERGEN_RELAX_MAX_SECONDS  (default 0 = off)
#       Wall-clock budget for the whole relaxation phase. When it expires, every
#       structure still in flight or not yet started is retired immediately. This
#       is the belt-and-braces guarantee that the job reaches the metrics phase
#       and writes its outputs, whatever the structures do. Set it to the job's
#       walltime minus enough for the metrics phase (~1 h per 10k structures).
#
# Retired structures are dropped from keep_idx, which evaluate.py turns into
# n_failed_jobs -- so they stay in the denominator of the reported metrics as
# failed jobs, exactly like the degenerate-cell skips above. They are NOT
# silently removed, and they are NOT reported as relaxed.
#
# A cap only changes results through structures that would have exceeded it. A
# run in which nothing is retired is bit-identical to an uncapped run, which is
# why the counts below are logged: they say precisely how far a capped run can
# differ from the uncapped ones it is compared against.
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
    """BatchRelaxer that retires structures instead of looping on them forever.

    `capped_indices` holds the indices (into the list passed to `relax`) of
    structures retired by either cap. Structures never started because the
    wall-clock budget expired first simply never appear in the returned
    trajectories, so callers should derive the successful set from
    `set(trajectories) - capped_indices`.
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

        Runs after step_batch(), which has already pruned the converged
        instances and reset is_active_instance to all-True, so everything left
        in optimizer_instances took a step this round and is still unconverged.
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
            # step_batch() set finished=False because these instances had not
            # converged. They are gone now, so recompute rather than deadlock.
            if not self.optimizer_instances:
                self.finished = True

    def relax(self, atoms_list: list[Atoms]) -> dict[int, list[Atoms]]:
        # Mirrors BatchRelaxer.relax, with the two caps added. Reimplemented
        # rather than wrapped because the deadline has to stop *insertion* too:
        # the outer loop keeps admitting new structures while any remain, so a
        # check placed only around step_batch would still let the job feed the
        # whole list through after the budget expired.
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
    # A few generated structures have a degenerate unit cell, in either
    # direction, and each direction is fatal to the whole run rather than to
    # the one structure -- BatchRelaxer packs structures by atom count alone, so
    # one bad sample takes down every structure batched with it.
    #
    # Collapsed: the relaxer builds a fixed-cutoff neighbor graph, so edges grow
    # with atom number density and three-body triplets with its square; a
    # collapsed cell blows the graph up over periodic images and exhausts GPU
    # memory.
    #
    # Exploded: pymatgen's find_points_in_spheres, which mattersim calls to
    # build that neighbor list, grids the cell into bins of roughly the cutoff
    # radius, so its allocation grows with *absolute* volume. A sample with a
    # 4e14 A^3 cell asks for ~3e12 bins and dies with
    # "MemoryError: Memory allocation of 21373999578960 bytes failed" partway
    # through the run, having relaxed nothing.
    #
    # Both thresholds are per atom, because density rather than absolute volume
    # is the physically meaningful quantity: a 20-atom cell of 20 A^3 is as
    # pathological as a 1-atom cell of 1 A^3. 2.0 A^3/atom sits well below the
    # densest physically real crystal (diamond, 5.7 A^3/atom). 1000 A^3/atom
    # sits far above the loosest: MP-20-OS itself spans 18 A^3/atom (median) to
    # 134 (max), with p99.9 at 88, so neither bound can fire on a plausible
    # sample. Given MP-20's 20-atom ceiling, the upper bound also caps absolute
    # volume at 2e4 A^3 -- about 160 neighbor-grid bins, against the ~3e12 above.
    #
    # Skipped structures are reported in keep_idx, which evaluate.py turns into
    # n_failed_jobs, so they count as failed jobs in the denominator of the
    # reported metrics rather than being dropped from it.
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
    max_steps = int(_env_number("MATTERGEN_RELAX_MAX_STEPS", 500))
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

    # Keep only structures that actually converged. Retired ones are dropped
    # from keep_idx so evaluate.py counts them as failed jobs; ones never
    # started (wall-clock budget) are absent from the trajectories entirely and
    # fall out here too. Sorting is a no-op when nothing is retired -- the
    # trajectory keys are inserted in ascending order -- so an uncapped run is
    # unchanged by this.
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
    """Relax structures using a machine-learning force field.

    Oxidation states are preserved across the relaxation: per-site OS values are
    captured before conversion to ASE (which strips them) and re-attached to the
    relaxed pymatgen structures afterwards. This allows the returned structures to
    be saved as OS-decorated CIFs for downstream analysis.

    Structures whose unit cell is degenerate -- collapsed to near-zero volume or
    inflated to an absurd one -- are skipped by `relax_atoms` (see the comment
    there) rather than relaxed, because either kind aborts the relaxation of
    every structure batched with it. The returned `keep_idx` gives the indices,
    into the input `structures`, of the structures actually relaxed -- callers
    must index any other per-structure list (e.g. the original structures) by
    the same `keep_idx` to stay aligned with `relaxed_structures`/`energies`.
    `evaluate.py` derives `n_failed_jobs` from it, so skipped structures remain
    in the denominator of the reported metrics.

    Parameters
    ----------
    structures
        One or more pymatgen structures to relax.
    device
        Torch device string for the MLFF potential.
    potential_load_path
        Path to a MatterSim checkpoint. None loads the default bundled model.
    output_path
        If provided, relaxed structures are written to this path in extxyz format.
    **kwargs
        Forwarded to BatchRelaxer.
    """
    if isinstance(structures, Structure):
        structures = [structures]

    # Capture per-site oxidation states before ASE conversion strips them.
    # Stored as None for structures that have plain Element (not Species) sites.
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

    # Re-attach oxidation states so downstream CIF output retains OS labels.
    # Index saved_os by keep_idx: relax_atoms may have skipped some structures,
    # so relaxed_structures no longer lines up 1:1 with the original ordering.
    for structure, os_list in zip(relaxed_structures, (saved_os[i] for i in keep_idx)):
        if os_list is not None:
            structure.add_oxidation_state_by_site(os_list)

    return relaxed_structures, total_energies, keep_idx
