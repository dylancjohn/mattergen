# OxiGen: Oxidation-State-Aware Crystal Generation

This directory contains OxiGen, an oxidation-state-aware crystal diffusion model. It is a modified
fork of [MatterGen](https://github.com/microsoft/mattergen)
([Zeni et al., 2025](https://www.nature.com/articles/s41586-025-08628-5)) and is distributed under
the original MIT licence (see [`LICENSE`](LICENSE)). It is not an official MatterGen release and is
not endorsed by the original authors.

Relative to upstream MatterGen, this fork adds:
* a species (element, oxidation state) vocabulary for atom types, and the MP-20-OS dataset
  labelled with oxidation states;
* MDLM masked discrete diffusion for atom types, alongside the original D3PM;
* structured charge-neutral sampling (and an optional structured training objective), built on
  the structured output layer in `neutral_layer.generation`;
* `mattergen-evaluate-os`, which computes the oxidation-state, validity and diversity metrics
  against MP-20-OS.

Installation is described in the top-level README. For anything not covered here, such as the
upstream pretrained checkpoints, reference datasets and other property conditioning, see the
[MatterGen repository](https://github.com/microsoft/mattergen).


## Data

MP-20-OS is included in [`datasets/mp-20-os`](datasets/mp-20-os) as `train`/`val`/`test` splits of
flat per-atom arrays, with an `oxidation_states.npy` and DFT band gaps (`dft_band_gap.json`) for
each split. It is MP-20 labelled with oxidation states by CrystaliteOS (see the top-level README).
Set `MP_20_OS_DATA_DIR` to use a copy elsewhere.


## Reproducing the paper

The commands below assume the working directory is this one. Training writes to `OUTPUT_DIR`
(default `outputs/singlerun/<date>/<time>`); that directory is then the `MODEL_PATH` for sampling
and fine-tuning. `~trainer.logger` disables Weights & Biases logging; remove it to log.
Generation is restricted to the same 76 elements as MatterGen (no Tc, Pm or Z >= 84).

### Models

| Paper name | Training config | Sampling config |
|---|---|---|
| **OxiGen** (species, structured sampling) | `species_mdlm` | `mdlm_constrained` |
| Ablation: element representation | `mdlm` | `mdlm` |
| Ablation: species, no structured output layer | `species_mdlm` | `mdlm` |
| Ablation: species, structured training and sampling | `species_mdlm_constrained` | `mdlm_constrained` |
| MatterGen baseline, retrained on MP-20-OS | `standard` | `default` |

OxiGen and the unconstrained species ablation are the same trained models, sampled with and
without the structured output layer.

### 1. Training

Each model is trained for five seeds (900 epochs, batch size 512; see the paper's appendix for all
settings), for example:

```bash
for SEED in 0 1 2 3 4; do
  OUTPUT_DIR=outputs/species_mdlm_seed${SEED} \
    mattergen-train --config-name=species_mdlm ~trainer.logger ++params.seed=${SEED}
done
```

Replace `species_mdlm` with any training config from the table. The batch size is split across
devices, so `trainer.devices`, `trainer.num_nodes` and `trainer.accumulate_grad_batches` can be
changed without changing the effective batch size.

### 2. Sampling

The paper uses 10,240 structures per seed, from the last-epoch checkpoint:

```bash
MODEL_PATH=outputs/species_mdlm_seed0
RESULTS_PATH=results/oxigen_seed0
mattergen-generate $RESULTS_PATH --model_path=$MODEL_PATH \
  --batch_size=128 --num_batches=80 --sampling_config_name=mdlm_constrained
```

This writes `generated_crystals_cif.zip` (one CIF per structure, with oxidation states for species
models) and `generated_crystals.extxyz`. The species vocabulary is detected from the checkpoint.
Without `--sampling_config_name`, the sampling config matching the training config is used, so it
is only needed for OxiGen: `species_mdlm` is trained without the structured output layer but
sampled with it.

### 3. Evaluation

The results use two evaluation scripts.

**Stability, uniqueness, novelty and relaxation** (`mattergen-evaluate`, unchanged from MatterGen).
Structures are relaxed with MatterSim and compared against MatterGen's Alex-MP reference dataset
with the MP2020 correction (see [`data-release/alex-mp`](data-release/alex-mp) for its sources and
licence). The reference dataset is stored with Git LFS and is not downloaded on clone:

```bash
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
mattergen-evaluate --structures_path=$RESULTS_PATH/generated_crystals.extxyz \
  --relax=True --structure_matcher=disordered \
  --save_as=$RESULTS_PATH/eval/metrics.json \
  --save_detailed_as=$RESULTS_PATH/eval/metrics_per_structure.json \
  --structures_output_path=$RESULTS_PATH/eval/relaxed_structures.extxyz
```

**Oxidation-state, validity and diversity metrics** (`mattergen-evaluate-os`). This scores the raw
(unrelaxed) CIFs against the MP-20-OS test split: compositional validity, the oxidation-state
frequency score and distance, and charge neutrality (all excluding alloys and single-element
structures), plus the CDVAE-style validity, coverage and Wasserstein metrics. CIFs that fail to
parse count as failures.

```bash
mattergen-evaluate-os $RESULTS_PATH/generated_crystals_cif.zip \
  --save_as=$RESULTS_PATH/eval/os_metrics.json
```

The MP-20-OS side is cached in `datasets/mp-20-os/test/evaluate_os_cache`. When evaluating many
runs in parallel, build the cache first with `mattergen-evaluate-os --build_cache_only`.
The Wasserstein distances use every valid structure by default; pass `--wasserstein_n_samples=1000`
to sample 1,000 as in CDVAE and DiffCSP. `--n_jobs` defaults to the CPUs allocated by Slurm or PBS.

### 4. Band-gap conditional generation

OxiGen and MatterGen are fine-tuned from their unconditional seed-0 last-epoch checkpoints on
MP-20-OS band gaps (200 epochs, batch size 128, learning rate 5e-6):

```bash
# OxiGen: species data. For MatterGen, use data_module=mp_bg_os and the `standard` checkpoint.
OUTPUT_DIR=outputs/species_mdlm_bg mattergen-finetune \
  adapter.model_path=outputs/species_mdlm_seed0 adapter.load_epoch=last \
  data_module=mp_bg_os_species \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
  data_module.properties='["dft_band_gap"]' ~trainer.logger
```

The diffusion process and denoiser settings are taken from the pretrained checkpoint. Sample
1,024 structures at each target band gap with classifier-free guidance strength 2:

```bash
for TARGET in 1.0 3.0 5.0 7.0; do
  mattergen-generate results_bg/oxigen/target_${TARGET} --model_path=outputs/species_mdlm_bg \
    --batch_size=128 --num_batches=8 --sampling_config_name=mdlm_constrained \
    --properties_to_condition_on="{'dft_band_gap': ${TARGET}}" --diffusion_guidance_factor=2.0
done
```

Evaluate each target with both scripts as above. The band gaps of the relaxed structures are
predicted with iComFormer ([Yan et al., 2024](https://github.com/divelab/AIRS)), which is not
included here.


## Options not used in the paper

These are available but were not used for any reported result:

* **D3PM on the species vocabulary**: `species` (unconstrained) and `species_constrained`
  (structured training) training configs, sampled with `default` or `d3pm_constrained`.
* **Duo** (uniform-state discrete diffusion): `duo` (element vocabulary), `species_duo` and
  `species_duo_constrained` training configs, sampled with `duo` or `duo_constrained`.
* The structured training objective with either family, via the `neutral_*` diffusion-module
  configs, which also expose the D3PM loss weights for ablation.

The upstream MatterGen features (pretrained checkpoints, other property conditioning, crystal
structure prediction and the Alex-MP-20 dataset) are unchanged; see the
[MatterGen README](https://github.com/microsoft/mattergen#readme).


## Citation

See [`CITATION.md`](CITATION.md).
