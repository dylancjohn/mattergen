"""End-to-end smoke tests for the charge-neutral MDLM and Duo configs.

Covers ``species_mdlm_constrained`` and ``species_duo_constrained`` without
the MP-20-OS dataset: each config's real GemNet model, corruption and
structured loss are built via Hydra and driven directly with a small
hand-built charge-neutral batch. This checks species embeddings at the right
vocabulary size, forward corruption, the DP-backed structured loss, backprop
through the whole model and checkpoint round-tripping.

Sampling is not exercised: GemNet's neighbour-graph construction raises on
the tiny (6-8 atom) structures a prior draw gives here, independently of
atom-type diffusion (the unconstrained ``species_mdlm`` config fails the same
way). The reverse steps these configs add are covered by
``test_neutral_mdlm_sampler.py`` and ``test_neutral_duo_sampler.py`` with
synthetic scores, through the same ``update_given_score`` interface the
predictor-corrector sampler uses.
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
