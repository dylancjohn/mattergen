"""Tests for inferring the sampling config from a checkpoint's training config.

Predictors' ``is_compatible()`` only rejects cross-family mismatches (e.g. an
MDLM predictor on a D3PM checkpoint). Constrained and unconstrained variants
share a corruption class, so sampling a constrained checkpoint with an
unconstrained config would silently drop charge neutrality. ``generate.py``
therefore picks the ``sampling_conf`` entrypoint from the saved corruption
and loss targets.
"""

import os

import hydra
import pytest

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.scripts.generate import _detect_sampling_config_name

CONFIG_DIR = os.path.join(MODELS_PROJECT_ROOT, "conf")


@pytest.mark.parametrize(
    "config_name,expected_sampling_config_name",
    [
        ("default", "default"),
        ("species_constrained", "d3pm_constrained"),
        ("mdlm", "mdlm"),
        ("species_mdlm_constrained", "mdlm_constrained"),
        ("duo", "duo"),
        ("species_duo_constrained", "duo_constrained"),
    ],
)
def test_detects_matching_sampling_config(
    config_name: str, expected_sampling_config_name: str
) -> None:
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR):
        config = hydra.compose(config_name=config_name)

    assert _detect_sampling_config_name(config) == expected_sampling_config_name


def test_returns_none_for_unrecognised_corruption_or_loss_targets():
    """Unrecognised setups (e.g. CSP) must return None rather than a guess."""
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "lightning_module": {
                "diffusion_module": {
                    "corruption": {
                        "discrete_corruptions": {"atomic_numbers": {"_target_": "some.other.Corruption"}}
                    },
                    "loss_fn": {"atomic_numbers_loss_partial": {"_target_": "some.other.loss"}},
                }
            }
        }
    )
    assert _detect_sampling_config_name(config) is None
