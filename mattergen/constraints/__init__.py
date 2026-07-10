"""constraints

MatterGen-specific adapters for oxidation-state-aware, charge-neutral generation.

Generic constraint machinery (DP, SPL, species vocabulary) lives in the
standalone ``neutral_layer`` package and is imported from there. This package
only contains the MatterGen data-pipeline glue that depends on MatterGen's own
classes (``ChemGraph``, ``BaseDataset``, Hydra/OmegaConf config composition).

Modules:
    resolvers   — OmegaConf ``species_vocab_size`` resolver for Hydra configs.
    dataset     — SpeciesCrystalDataset for loading species-indexed crystal data.
    datamodule  — SpeciesDataModule, a Hydra-instantiable LightningDataModule.
"""
