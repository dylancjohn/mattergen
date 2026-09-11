# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from ase import Atoms
from ase.io import write
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from mattersim.utils.logger_utils import get_logger
from pymatgen.core import Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.utils.globals import get_device

logger = get_logger()
logger.level("ERROR")


def relax_atoms(
    atoms: list[Atoms], device: str = str(get_device()), potential_load_path: str = None, output_path: str | None = None, **kwargs
) -> tuple[list[Atoms], np.ndarray, list[int]]:
    # A few generated structures can have a collapsed unit cell. The relaxer
    # builds a fixed-cutoff neighbor graph, so edges grow with atom number
    # density and three-body triplets with its square; a collapsed cell blows
    # the graph up over periodic images and exhausts GPU memory. Because
    # BatchRelaxer packs structures by atom count alone, one such structure
    # takes down every structure batched with it. Skip them here instead, and
    # count them as failed jobs rather than dropping them from the metrics.
    #
    # The threshold is per atom because density, not absolute volume, drives
    # graph size: a 20-atom cell of 20 A^3 is as pathological as a 1-atom cell
    # of 1 A^3. 2.0 A^3/atom sits well below the densest physically real
    # crystal (diamond, 5.7 A^3/atom), so it only fires on degenerate samples.
    MIN_VOLUME_PER_ATOM_A3 = 2.0
    keep_idx = [
        i
        for i, a in enumerate(atoms)
        if len(a) > 0 and a.get_volume() / len(a) >= MIN_VOLUME_PER_ATOM_A3
    ]
    if len(keep_idx) < len(atoms):
        logger.warning(
            f"Skipping relaxation for {len(atoms) - len(keep_idx)} structure(s) "
            f"with unit cell volume < {MIN_VOLUME_PER_ATOM_A3} A^3 per atom"
        )

    potential = Potential.from_checkpoint(
        device=device, load_path=potential_load_path, load_training_state=False
    )
    batch_relaxer = BatchRelaxer(potential=potential, filter="EXPCELLFILTER", **kwargs)
    relaxation_trajectories = batch_relaxer.relax([atoms[i] for i in keep_idx])
    relaxed_atoms = [t[-1] for t in relaxation_trajectories.values()]
    total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])
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

    Structures with a degenerate unit cell are skipped by `relax_atoms` (see its
    docstring) rather than relaxed. The returned `keep_idx` gives the indices,
    into the input `structures`, of the structures actually relaxed -- callers
    must index any other per-structure list (e.g. the original structures) by
    the same `keep_idx` to stay aligned with `relaxed_structures`/`energies`.

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
