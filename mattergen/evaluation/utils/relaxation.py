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
) -> tuple[list[Atoms], np.ndarray]:
    potential = Potential.from_checkpoint(
        device=device, load_path=potential_load_path, load_training_state=False
    )
    batch_relaxer = BatchRelaxer(potential=potential, filter="EXPCELLFILTER", **kwargs)
    relaxation_trajectories = batch_relaxer.relax(atoms)
    relaxed_atoms = [t[-1] for t in relaxation_trajectories.values()]
    total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])
    if output_path:
        write(output_path, relaxed_atoms, format="extxyz")
        logger.info(f"Relaxed structures saved to {output_path}")
    return relaxed_atoms, total_energies


def relax_structures(
    structures: Structure | list[Structure],
    device: str = str(get_device()),
    potential_load_path: str = None,
    output_path: str | None = None,
    **kwargs,
) -> tuple[list[Structure], np.ndarray]:
    """Relax structures using a machine-learning force field.

    Oxidation states are preserved across the relaxation: per-site OS values are
    captured before conversion to ASE (which strips them) and re-attached to the
    relaxed pymatgen structures afterwards. This allows the returned structures to
    be saved as OS-decorated CIFs for downstream analysis.

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
    relaxed_atoms, total_energies = relax_atoms(
        atoms, device=device, potential_load_path=potential_load_path, output_path=output_path, **kwargs
    )
    relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]

    # Re-attach oxidation states so downstream CIF output retains OS labels.
    for structure, os_list in zip(relaxed_structures, saved_os):
        if os_list is not None:
            structure.add_oxidation_state_by_site(os_list)

    return relaxed_structures, total_energies
