# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from pathlib import Path
from typing import Literal

import fire
from omegaconf import OmegaConf
from pymatgen.core.structure import Structure

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME, MatterGenCheckpointInfo, ProgressCallback
from mattergen.generator import CrystalGenerator

logger = logging.getLogger(__name__)

_SPECIES_DATAMODULE_TARGET = "mattergen.common.data.species_datamodule.SpeciesDataModule"

_FAMILY_BY_CORRUPTION_TARGET = {
    "mattergen.diffusion.corruption.d3pm_corruption.D3PMCorruption": "d3pm",
    "mattergen.diffusion.corruption.mdlm_corruption.MDLMCorruption": "mdlm",
    "mattergen.diffusion.corruption.duo_corruption.DuoCorruption": "duo",
}
_CONSTRAINED_BY_LOSS_TARGET = {
    "mattergen.diffusion.d3pm.d3pm_loss.d3pm_loss": False,
    "mattergen.diffusion.d3pm.neutral_d3pm_loss.make_neutral_d3pm_loss": True,
    "mattergen.diffusion.mdlm.mdlm_loss.mdlm_loss": False,
    "mattergen.diffusion.mdlm.neutral_mdlm_loss.make_neutral_mdlm_loss": True,
    "mattergen.diffusion.duo.duo_loss.duo_loss": False,
    "mattergen.diffusion.duo.neutral_duo_loss.make_neutral_duo_loss": True,
}
_SAMPLING_CONFIG_NAME_BY_FAMILY_AND_CONSTRAINT = {
    ("d3pm", False): "default",
    ("d3pm", True): "d3pm_constrained",
    ("mdlm", False): "mdlm",
    ("mdlm", True): "mdlm_constrained",
    ("duo", False): "duo",
    ("duo", True): "duo_constrained",
}


def _detect_sampling_config_name(cfg) -> str | None:
    """Infer the sampling_conf entrypoint matching this checkpoint's trained
    atom-type family and constraint mode, from its own saved training config.

    Returns None if the corruption/loss targets aren't recognised (e.g.
    non-atom-type-diffusion setups like CSP), so callers can fall back to
    prior behaviour rather than guessing.
    """
    corruption_target = OmegaConf.select(
        cfg,
        "lightning_module.diffusion_module.corruption.discrete_corruptions.atomic_numbers._target_",
        default=None,
    )
    loss_target = OmegaConf.select(
        cfg,
        "lightning_module.diffusion_module.loss_fn.atomic_numbers_loss_partial._target_",
        default=None,
    )
    family = _FAMILY_BY_CORRUPTION_TARGET.get(corruption_target)
    constrained = _CONSTRAINED_BY_LOSS_TARGET.get(loss_target)
    if family is None or constrained is None:
        return None
    return _SAMPLING_CONFIG_NAME_BY_FAMILY_AND_CONSTRAINT[(family, constrained)]


def main(
    output_path: str,
    pretrained_name: PRETRAINED_MODEL_NAME | None = None,
    model_path: str | None = None,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str | None = None,
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
    progress_callback: ProgressCallback | None = None,
    use_species_vocab: bool | None = None,
) -> list[Structure]:
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). Defaults to None, which auto-detects the config matching the checkpoint's trained atom-type family and constraint mode (falling back to "default" if that can't be determined, e.g. for non-atom-type-diffusion setups like CSP). Only pass this explicitly to override detection -- e.g. to intentionally sample a constrained/unconstrained checkpoint with the other's config.
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.
        target_compositions: List of dictionaries with target compositions to condition on. Each dictionary should have the form `{element: number_of_atoms}`. If None, the target compositions are not conditioned on.
           Only supported for models trained for crystal structure prediction (CSP) (default: None)
        progress_callback: Optional callback function that takes in a single float argument representing the progress of the generation process (between 0 and 1).
        use_species_vocab: When True, decode generated atom types as species vocab indices (element + oxidation state) rather than plain atomic numbers. Defaults to None, which auto-detects from the checkpoint's own saved data_module config (whether it was trained with SpeciesDataModule). Only pass this explicitly to override detection.
    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."
    assert (
        pretrained_name is None or model_path is None
    ), "Only one of pretrained_name or model_path can be provided."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    if pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            pretrained_name, config_overrides=config_overrides
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading,
        )

    # Auto-detect whether this checkpoint was trained with the species vocabulary
    # (element + oxidation state) from its own saved data_module config, rather than
    # trusting a caller-supplied flag: use_species_vocab previously defaulted to
    # False, so a forgotten flag would silently install the wrong allow-list and
    # decode species indices as plain atomic numbers instead of failing loudly.
    detected_species_vocab = (
        OmegaConf.select(checkpoint_info.config, "data_module._target_", default=None)
        == _SPECIES_DATAMODULE_TARGET
    )
    if use_species_vocab is None:
        use_species_vocab = detected_species_vocab
    elif use_species_vocab != detected_species_vocab:
        logger.warning(
            "use_species_vocab=%s was explicitly passed, but the checkpoint's own "
            "data_module config indicates use_species_vocab=%s. Proceeding with the "
            "explicit value, but this mismatch usually means a mistake.",
            use_species_vocab,
            detected_species_vocab,
        )

    # Disable generating element types which are not supported or not in the desired chemical
    # system (if provided). mask_disallowed_elements operates over the element vocab; species
    # models use mask_disallowed_species instead, which applies the same SELECTED_ATOMIC_NUMBERS
    # allow-list over species indices (derived from the vocabulary at runtime).
    if not use_species_vocab:
        checkpoint_info.config_overrides.append(
            "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
        )
    else:
        checkpoint_info.config_overrides.append(
            "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_species',_partial_:True}"
        )
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None

    # Auto-detect the sampling_conf entrypoint matching this checkpoint's trained
    # atom-type family and constraint mode, from its own saved training config.
    # is_compatible() on the predictors only rejects cross-family mismatches (e.g.
    # MDLM predictor on a D3PM checkpoint); it does not catch constrained-vs-
    # unconstrained mismatches, since corruption classes are shared identically
    # between a family's constrained and unconstrained variants. In particular,
    # sampling a constrained-trained checkpoint with an unconstrained config
    # silently drops the charge-neutrality guarantee with no error at all.
    detected_sampling_config_name = _detect_sampling_config_name(checkpoint_info.config)
    if sampling_config_name is None:
        sampling_config_name = detected_sampling_config_name or "default"
    elif (
        detected_sampling_config_name is not None
        and sampling_config_name != detected_sampling_config_name
    ):
        logger.warning(
            "sampling_config_name=%r was explicitly passed, but the checkpoint's own "
            "training config indicates sampling_config_name=%r (family/constraint mode "
            "detected from its corruption and loss targets). Proceeding with the "
            "explicit value, but this mismatch usually means a mistake -- in particular, "
            "sampling a constrained-trained checkpoint with an unconstrained config "
            "silently drops the charge-neutrality guarantee.",
            sampling_config_name,
            detected_sampling_config_name,
        )

    species_vocab = None
    if use_species_vocab:
        from neutral_layer.data.vocab import build_species_vocab

        species_vocab = build_species_vocab()

    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
        progress_callback=progress_callback,
        species_vocab=species_vocab,
    )
    return generator.generate(output_dir=Path(output_path))


def _main():
    # use fire instead of argparse to allow for the specification of dictionary values via the CLI
    fire.Fire(main)


if __name__ == "__main__":
    _main()
