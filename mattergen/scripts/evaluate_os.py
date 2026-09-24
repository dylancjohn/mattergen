"""Command-line entry point for `mattergen.evaluation.evaluate_os`."""

import json
from pathlib import Path

import fire

from mattergen.evaluation.evaluate_os import (
    build_reference_cache,
    default_mp20_os_dir,
    evaluate_os,
    load_cif_structures,
)
from mattergen.evaluation.reference.mp20_os import load_mp20_os_reference_dataset
from mattergen.evaluation.utils.parallel import default_n_jobs


def main(
    structures_path: str | None = None,
    save_as: str | None = None,
    mp20_os_dir: str | None = None,
    split: str = "test",
    cache_dir: str | None = None,
    no_cache: bool = False,
    n_jobs: int | None = None,
    consensus: int = 3,
    struc_cutoff: float = 0.4,
    comp_cutoff: float = 10.0,
    wasserstein_n_samples: int = 0,
    wasserstein_seed: int = 0,
    exclude_nonchargeable: bool = True,
    build_cache_only: bool = False,
):
    """Evaluate raw generated structures against an MP-20-OS split.

    `structures_path` is a zip or directory of the generated CIFs (e.g.
    `generated_crystals_cif.zip`); structures should not be relaxed. The
    reference-side cache defaults to `<mp20_os_dir>/<split>/evaluate_os_cache`.
    With `build_cache_only`, only that cache is built and no structures are
    needed. `wasserstein_n_samples=0` uses every valid structure; pass 1000 to
    reproduce the CDVAE/DiffCSP protocol.
    """
    mp20_os_path = Path(mp20_os_dir) if mp20_os_dir else default_mp20_os_dir()
    cache_path = None
    if not no_cache:
        cache_path = Path(cache_dir) if cache_dir else mp20_os_path / split / "evaluate_os_cache"
    n_jobs = n_jobs if n_jobs is not None else default_n_jobs()
    reference = load_mp20_os_reference_dataset(mp20_os_path, split)

    if build_cache_only:
        if cache_path is None:
            raise ValueError("build_cache_only needs a cache; do not pass no_cache.")
        build_reference_cache(reference, cache_path, consensus=consensus, n_jobs=n_jobs)
        return
    if structures_path is None:
        raise ValueError("structures_path is required unless build_cache_only is set.")

    structures, n_parse_failures = load_cif_structures(structures_path, n_jobs=n_jobs)
    metrics = evaluate_os(
        structures,
        reference,
        n_parse_failures=n_parse_failures,
        cache_dir=cache_path,
        n_jobs=n_jobs,
        consensus=consensus,
        struc_cutoff=struc_cutoff,
        comp_cutoff=comp_cutoff,
        wasserstein_n_samples=wasserstein_n_samples,
        wasserstein_seed=wasserstein_seed,
        exclude_nonchargeable=exclude_nonchargeable,
    )
    metrics["_metadata"] = {
        "reference": reference.name,
        "n_reference": len(reference),
        "consensus": consensus,
        "struc_cutoff": struc_cutoff,
        "comp_cutoff": comp_cutoff,
        "wasserstein_n_samples": wasserstein_n_samples,
        "wasserstein_seed": wasserstein_seed,
        "exclude_nonchargeable": exclude_nonchargeable,
    }
    if save_as is not None:
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        Path(save_as).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
