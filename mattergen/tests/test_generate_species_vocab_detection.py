# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test_generate_species_vocab_detection.py

Tests for generate.py's checkpoint-driven species-vocab auto-detection: the
config's data_module._target_ must reliably distinguish species-vocab
checkpoints (SpeciesDataModule) from element-vocab ones (CrystDataModule),
since generate.main() uses exactly this signal to decide which allow-list
masking function and vocab decoder to install, rather than trusting a
caller-supplied flag that could silently mis-decode a forgotten case.
"""

import os

import hydra
import pytest
from omegaconf import OmegaConf

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.scripts.generate import _SPECIES_DATAMODULE_TARGET

CONFIG_DIR = os.path.join(MODELS_PROJECT_ROOT, "conf")


@pytest.mark.parametrize(
    "config_name,expected_species_vocab",
    [
        ("default", False),
        ("mdlm", False),
        ("duo", False),
        ("species_constrained", True),
        ("species_mdlm_constrained", True),
        ("species_duo_constrained", True),
    ],
)
def test_data_module_target_detects_species_vocab(
    config_name: str, expected_species_vocab: bool
) -> None:
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR):
        config = hydra.compose(config_name=config_name)

    detected = (
        OmegaConf.select(config, "data_module._target_", default=None)
        == _SPECIES_DATAMODULE_TARGET
    )
    assert detected == expected_species_vocab
