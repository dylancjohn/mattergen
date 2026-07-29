# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end smoke tests for the two new structured (charge-neutral) configs,
``species_mdlm_constrained`` and ``species_duo_constrained``.

These do not use ``mattergen_main`` / a real ``DataLoader``: the MP-20-OS
cached dataset this cluster's ``data_module: mp_20_species`` config points at
is not present in this environment (confirmed by ``test_train_on_one_batch``
failing identically for the pre-existing, unrelated ``default`` config with a
missing-file error), so no config's full train pipeline can be exercised here.
Instead, this builds each config's *real* GemNet-based model, corruption and
constrained loss via Hydra, and drives them directly with a small
hand-constructed but genuinely charge-neutral batch. This exercises
everything the missing dataset would otherwise supply: real species
embeddings at the correct vocab size, real forward corruption, the real
constrained DP-backed loss, backprop through the whole model, and checkpoint
round-tripping.

Sampling (``PredictorCorrector.sample``) is deliberately not exercised here:
GemNet's neighbour-graph construction (``gemnet.py::generate_interaction_graph``)
raises on the tiny (6-8 atom) synthetic structures a from-scratch prior draw
produces in this setup, in code that has nothing to do with atom-type
diffusion. The same failure was confirmed to occur identically for the
pre-existing, unconstrained ``species_mdlm`` config with the same synthetic
conditioning batch, so it is a pre-existing GemNet/small-structure limitation,
not a regression from this work. The reverse-step logic these configs add
(``NeutralMDLMAncestralSamplingPredictor``/``NeutralDuoAncestralSamplingPredictor``
-- reveal probabilities, joint neutral sampling, exact neutrality) is already
directly and thoroughly covered in ``test_neutral_mdlm_sampler.py`` /
``test_neutral_duo_sampler.py`` using synthetic score tensors, which is the
same call contract the PC sampler uses (``update_given_score(x=..., t=...,
dt=..., batch_idx=..., score=..., batch=...)``) without requiring GemNet.
"""

from __future__ import annotations

import copy
import os

import hydra
import pytest
import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.constraints.charges import build_charge_of
from mattergen.scripts.run import (
    mattergen_main,  # noqa: F401  (registers OmegaConf resolvers)
)

CONFIG_DIR = os.path.join(MODELS_PROJECT_ROOT, "conf")

_SIZE_OVERRIDES = [
    "lightning_module.diffusion_module.model.gemnet.num_blocks=1",
    "lightning_module.diffusion_module.model.hidden_dim=16",
]


def _neutral_species_pair() -> tuple[int, int]:
    """Two distinct 0-based species indices whose charges sum to zero."""
    charge_of, mask_idx = build_charge_of()
    for i in range(mask_idx):
        for j in range(mask_idx):
            if i != j and charge_of[i] + charge_of[j] == 0:
                return i, j
    raise AssertionError("expected at least one oppositely-charged species pair")


def _synthetic_neutral_batch() -> ChemGraph:
    """Two small charge-neutral crystals built from real species-vocab charges."""
    a, b = _neutral_species_pair()
    torch.manual_seed(0)
    samples = [
        ChemGraph(
            pos=torch.rand(4, 3),
            cell=torch.eye(3, dtype=torch.float).unsqueeze(0) * 8.0,
            atomic_numbers=torch.tensor([a, b, a, b], dtype=torch.long) + 1,  # 1-based
            num_atoms=torch.tensor([4], dtype=torch.long),
            num_nodes=4,
        ),
        ChemGraph(
            pos=torch.rand(6, 3),
            cell=torch.eye(3, dtype=torch.float).unsqueeze(0) * 8.0,
            atomic_numbers=torch.tensor([b, a, b, a, b, a], dtype=torch.long) + 1,
            num_atoms=torch.tensor([6], dtype=torch.long),
            num_nodes=6,
        ),
    ]
    return collate(samples)


@pytest.mark.parametrize(
    "config_name", ["species_mdlm_constrained", "species_duo_constrained"]
)
def test_constrained_config_trains_one_step_with_finite_loss(config_name: str) -> None:
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR):
        config = hydra.compose(config_name=config_name, overrides=_SIZE_OVERRIDES)

    lightning_module = hydra.utils.instantiate(config.lightning_module)
    diffusion_module = lightning_module.diffusion_module

    batch = _synthetic_neutral_batch()

    optimizer = torch.optim.SGD(diffusion_module.parameters(), lr=1e-3)
    params_before = [p.detach().clone() for p in diffusion_module.parameters()]

    loss, metrics = diffusion_module.calc_loss(batch)
    assert torch.isfinite(loss).all(), f"non-finite loss: {loss}"
    for name, value in metrics.items():
        assert torch.isfinite(value).all(), f"non-finite metric {name}: {value}"

    optimizer.zero_grad()
    loss.backward()
    for p in diffusion_module.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient after backward"
    optimizer.step()

    params_after = list(diffusion_module.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    ), "optimizer step did not change any parameters"


@pytest.mark.parametrize(
    "config_name", ["species_mdlm_constrained", "species_duo_constrained"]
)
def test_constrained_config_checkpoint_round_trips(config_name: str) -> None:
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR):
        config = hydra.compose(config_name=config_name, overrides=_SIZE_OVERRIDES)

    diffusion_module = hydra.utils.instantiate(config.lightning_module).diffusion_module
    state_dict = copy.deepcopy(diffusion_module.state_dict())

    reloaded = hydra.utils.instantiate(config.lightning_module).diffusion_module
    reloaded.load_state_dict(state_dict)

    batch = _synthetic_neutral_batch()
    torch.manual_seed(0)
    loss_original, _ = diffusion_module.calc_loss(batch)
    torch.manual_seed(0)
    loss_reloaded, _ = reloaded.calc_loss(batch)
    assert torch.allclose(loss_original.detach(), loss_reloaded.detach(), atol=1e-5)
