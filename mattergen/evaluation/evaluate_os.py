"""Oxidation-state, validity and diversity metrics against an MP-20-OS split.

`evaluate_os` scores raw (unrelaxed) generated structures, following the
CDVAE/DiffCSP protocol, which computes these metrics on model samples. They are
read from CIFs so that per-site oxidation states are available. The metrics are:

* validity: ``comp_valid``, ``struct_valid`` and ``valid`` (CDVAE/DiffCSP
  definitions over all structures, where ``valid`` also requires a structure
  fingerprint) and ``comp_valid_non_alloy_or_single_element`` (excluding
  alloys and single-element structures, and without SMACT's alloy shortcut);
* oxidation states: ``frac_alloy_or_single_element``, ``frac_charge_neutral``,
  ``avg_oxidation_state_frequency_score`` and ``oxidation_state_distance``,
  with alloys and single-element structures excluded;
* diversity: ``cov_recall``, ``cov_precision``, ``wdist_density`` and
  ``wdist_num_elems``. As in DiffCSP, coverage uses every structure and the
  Wasserstein distances only valid ones.

CIFs that fail to parse are counted as failures: they are invalid, not charge
neutral and unmatched, and stay in the denominator of every fraction. Averages
and distributions (frequency score, OS distance, Wasserstein distances) cover
parsed structures only.

The reference side of each metric depends only on the MP-20-OS split, so it is
cached to disk and shared across runs.
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import numpy as np
from pymatgen.core import Structure
from scipy.stats import wasserstein_distance

from mattergen.common.utils.eval_utils import _cif_sort_key
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.fingerprints import (
    composition_fingerprint,
    compute_coverage,
    structure_fingerprint,
)
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.oxidation_states import (
    aggregate_marginalized_distribution,
    composition_symbols_and_counts,
    is_alloy_or_single_element,
    marginalized_occurrences,
    mean_js_distance,
    oxidation_state_frequency_score,
)
from mattergen.evaluation.utils.parallel import parallel_map, parallel_map_with_stall_timeout

CompositionKey = tuple[tuple[str, ...], tuple[int, ...]]

# Bump if the cache layout changes; a mismatch is treated as a cache miss.
_CACHE_FORMAT_VERSION = 1


def default_mp20_os_dir() -> Path:
    """MP-20-OS location, as in the ``mp_20_os`` data module config."""
    return Path(
        os.environ.get("MP_20_OS_DATA_DIR", MODELS_PROJECT_ROOT.parent / "datasets" / "mp-20-os")
    )


# -----------------------------#
# Loading
# -----------------------------#


def _parse_cif(text: str) -> Structure | str:
    """The parsed structure, or the error message if parsing fails."""
    try:
        return Structure.from_str(text, fmt="cif")
    # pymatgen raises many exception types on malformed CIFs; the failure is
    # returned, then logged and counted by the caller rather than ignored.
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def load_cif_structures(
    path: str | Path, n_jobs: int | None = None
) -> tuple[list[Structure], int]:
    """Parse the CIFs in a zip archive or directory, in ``gen_{i}`` order.

    Returns ``(structures, n_parse_failures)``. Oxidation states on the sites
    are kept.
    """
    path = Path(path)
    if path.suffix == ".zip":
        with ZipFile(path) as zf:
            names = sorted(
                (n for n in zf.namelist() if n.endswith(".cif")),
                key=lambda n: _cif_sort_key(Path(n).name),
            )
            texts = [zf.read(n).decode("utf-8") for n in names]
    elif path.is_dir():
        names = sorted((p.name for p in path.iterdir() if p.suffix == ".cif"), key=_cif_sort_key)
        texts = [(path / n).read_text() for n in names]
    else:
        raise ValueError(f"Expected a .zip of CIFs or a directory of CIFs, got {path}")
    if not texts:
        raise ValueError(f"No CIF files found in {path}")

    structures: list[Structure] = []
    n_failed = 0
    for name, result in zip(names, parallel_map(_parse_cif, texts, n_jobs)):
        if isinstance(result, Structure):
            structures.append(result)
        else:
            n_failed += 1
            logger.warning(f"Failed to parse {name}: {result}")
    if n_failed:
        logger.warning(f"{n_failed} of {len(texts)} CIFs failed to parse; counted as failures.")
    return structures, n_failed


def _oxidation_state_labels(structure: Structure) -> list[int] | None:
    """Per-site oxidation states, or None if any site has none."""
    labels = []
    for site in structure:
        oxi_state = getattr(site.specie, "oxi_state", None)
        if oxi_state is None:
            return None
        labels.append(round(oxi_state))
    return labels


def _strip_oxidation_states(structures: Sequence[Structure]) -> list[Structure]:
    # The reference has plain Element sites. Species sites would count a
    # mixed-valence element twice in `wdist_num_elems`, and change CrystalNN's
    # neighbour assignment and hence the structure fingerprints.
    stripped = []
    for structure in structures:
        copy = structure.copy()
        copy.remove_oxidation_states()
        stripped.append(copy)
    return stripped


# -----------------------------#
# Reference side (cached)
# -----------------------------#


def _fingerprint_cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "reference_fingerprints.npz", cache_dir / "reference_fingerprints_meta.json"


def reference_fingerprints(
    reference: ReferenceDataset, cache_dir: Path | None, n_jobs: int | None
) -> tuple[list[np.ndarray], list[np.ndarray | None], np.ndarray, np.ndarray]:
    """Reference ``(comp_fps, struct_fps, densities, num_elements)``, cached in ``cache_dir``."""
    structures = [entry.structure for entry in reference]
    n_entries = len(structures)

    if cache_dir is not None:
        npz_path, meta_path = _fingerprint_cache_paths(cache_dir)
        if npz_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("format_version") == _CACHE_FORMAT_VERSION and meta.get("n_entries") == n_entries:
                data = np.load(npz_path)
                struct_fps = [
                    row if valid else None
                    for row, valid in zip(data["struct_fps"], data["struct_fp_valid"])
                ]
                logger.info(f"Loaded reference fingerprints from {npz_path}")
                return list(data["comp_fps"]), struct_fps, data["densities"], data["num_elements"]
            logger.warning(f"Reference fingerprint cache in {cache_dir} is stale; recomputing.")

    logger.info(f"Fingerprinting {n_entries} reference structures")
    comp_fps = parallel_map(composition_fingerprint, [s.composition for s in structures], n_jobs)
    struct_fps = parallel_map(structure_fingerprint, structures, n_jobs)
    densities = np.array([s.density for s in structures])
    num_elements = np.array([len(set(s.species)) for s in structures])

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path, meta_path = _fingerprint_cache_paths(cache_dir)
        # Missing structure fingerprints are stored as zero rows and restored to
        # None from `struct_fp_valid` on load.
        dim = next((len(fp) for fp in struct_fps if fp is not None), 0)
        struct_arr = np.zeros((n_entries, dim))
        for i, fp in enumerate(struct_fps):
            if fp is not None:
                struct_arr[i] = fp
        np.savez_compressed(
            npz_path,
            comp_fps=np.array(comp_fps, dtype=np.float64),
            struct_fps=struct_arr,
            struct_fp_valid=np.array([fp is not None for fp in struct_fps]),
            densities=densities,
            num_elements=num_elements,
        )
        meta_path.write_text(
            json.dumps({"format_version": _CACHE_FORMAT_VERSION, "n_entries": n_entries})
        )
        logger.info(f"Wrote reference fingerprints to {npz_path}")

    return comp_fps, struct_fps, densities, num_elements


def reference_oxidation_state_distribution(
    reference: ReferenceDataset, consensus: int, cache_dir: Path | None, n_jobs: int | None
) -> dict[str, dict[int, float]]:
    """Marginalised reference oxidation-state distribution, cached in ``cache_dir``."""
    compositions = [composition_symbols_and_counts(entry.structure) for entry in reference]
    n_entries = len(compositions)

    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"reference_oxstate_dist_consensus{consensus}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("format_version") == _CACHE_FORMAT_VERSION
                and cached.get("n_entries") == n_entries
                and cached.get("consensus") == consensus
            ):
                logger.info(f"Loaded reference oxidation-state distribution from {cache_path}")
                # JSON keys are strings; mean_js_distance needs int oxidation states.
                return {
                    elem: {int(ox_state): p for ox_state, p in dist.items()}
                    for elem, dist in cached["distribution"].items()
                }
            logger.warning(f"Reference oxidation-state cache {cache_path} is stale; recomputing.")

    logger.info(f"Marginalising oxidation states of {n_entries} reference structures")
    distribution = _marginalized_distribution(compositions, consensus, n_jobs)

    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "format_version": _CACHE_FORMAT_VERSION,
                    "n_entries": n_entries,
                    "consensus": consensus,
                    "distribution": distribution,
                }
            )
        )
        logger.info(f"Wrote reference oxidation-state distribution to {cache_path}")
    return distribution


def build_reference_cache(
    reference: ReferenceDataset, cache_dir: Path, consensus: int = 3, n_jobs: int | None = None
) -> None:
    """Fill ``cache_dir`` up front, so that parallel runs share one cache."""
    reference_fingerprints(reference, cache_dir, n_jobs)
    reference_oxidation_state_distribution(reference, consensus, cache_dir, n_jobs)


# -----------------------------#
# Metrics
# -----------------------------#


# The admissible-assignment search is memoized per process, which a process pool
# cannot share, so each unique composition is computed once instead.
def _frequency_score_for(composition: CompositionKey, consensus: int) -> float:
    return oxidation_state_frequency_score(*composition, consensus)


def _marginalized_for(
    composition: CompositionKey, consensus: int
) -> dict[str, dict[int, float]] | None:
    return marginalized_occurrences(*composition, consensus)


def _map_unique(func, compositions: Sequence[CompositionKey], n_jobs: int | None) -> dict:
    unique = list(dict.fromkeys(compositions))
    return dict(zip(unique, parallel_map(func, unique, n_jobs)))


def _marginalized_distribution(
    compositions: Sequence[CompositionKey], consensus: int, n_jobs: int | None
) -> dict[str, dict[int, float]]:
    """Dataset-level marginalised distribution; alloys contribute nothing."""
    non_alloy = [c for c in compositions if not is_alloy_or_single_element(c[0])]
    occurrences = _map_unique(partial(_marginalized_for, consensus=consensus), non_alloy, n_jobs)
    return aggregate_marginalized_distribution(occurrences[c] for c in non_alloy)


def _validity_flags(structure: Structure) -> tuple[bool, bool, bool, bool]:
    """``(comp_valid, charge_balanced, struct_valid, unknown_element)`` for one structure.

    ``charge_balanced`` is SMACT validity without its alloy shortcut. SMACT
    raises ``TypeError`` for elements it has no data for; such a composition
    is counted as invalid and reported by the caller.
    """
    struct_valid = bool(structure_validity(structure))
    try:
        return (
            bool(is_smact_valid(structure)),
            bool(is_smact_valid(structure, include_alloys=False)),
            struct_valid,
            False,
        )
    except TypeError:
        return False, False, struct_valid, True


def _frac_charge_neutral(
    labels: Sequence[list[int] | None], alloy_mask: np.ndarray, n_parse_failures: int
) -> float:
    """Fraction of labelled non-alloy structures whose oxidation states sum to 0.

    NaN if no structure carries oxidation states (e.g. an element-vocabulary
    model). Parse failures count as not neutral.
    """
    if all(label is None for label in labels):
        return float("nan")
    sums = [sum(label) for label, alloy in zip(labels, alloy_mask) if label is not None and not alloy]
    n_total = len(sums) + n_parse_failures
    return sum(s == 0 for s in sums) / n_total if n_total else float("nan")


def _wasserstein_distances(
    structures: Sequence[Structure],
    valid: np.ndarray,
    ref_densities: np.ndarray,
    ref_num_elements: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    """Density and N-ary Wasserstein distances over valid structures.

    ``n_samples <= 0`` uses every valid structure, which avoids the sampling
    noise of DiffCSP's n=1000 draw. Pass 1000 for directly comparable numbers.
    """
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        logger.warning("No valid generated structures; Wasserstein distances are NaN.")
        return float("nan"), float("nan")
    if 0 < n_samples <= len(valid_indices):
        valid_indices = np.random.RandomState(seed).choice(valid_indices, n_samples, replace=False)
    elif n_samples > 0:
        # DiffCSP raises here; this run's n then differs from the n=1000 protocol.
        logger.warning(
            f"Only {len(valid_indices)} valid structures, fewer than n_samples={n_samples}; "
            "using all of them."
        )
    densities = [structures[i].density for i in valid_indices]
    num_elements = [len(set(structures[i].species)) for i in valid_indices]
    return (
        float(wasserstein_distance(densities, ref_densities)),
        float(wasserstein_distance(num_elements, ref_num_elements)),
    )


def evaluate_os(
    structures: Sequence[Structure],
    reference: ReferenceDataset,
    n_parse_failures: int = 0,
    cache_dir: Path | None = None,
    n_jobs: int | None = None,
    consensus: int = 3,
    struc_cutoff: float = 0.4,
    comp_cutoff: float = 10.0,
    wasserstein_n_samples: int = 0,
    wasserstein_seed: int = 0,
    exclude_nonchargeable: bool = True,
) -> dict[str, float | int]:
    """Compute the module's metrics for ``structures`` against ``reference``.

    ``structures`` are the parsed generated structures, with any oxidation
    states still on their sites; ``n_parse_failures`` generated structures
    could not be parsed. ``consensus`` is the minimum ICSD24 occurrence count of
    an admissible oxidation state. With ``exclude_nonchargeable``, compositions
    with no admissible assignment are left out of the frequency-score average
    (they score 0; compositional validity already captures them).
    """
    labels = [_oxidation_state_labels(s) for s in structures]
    structures = _strip_oxidation_states(structures)
    n_total = len(structures) + n_parse_failures

    compositions = [composition_symbols_and_counts(s) for s in structures]
    alloy_mask = np.array([is_alloy_or_single_element(c[0]) for c in compositions], dtype=bool)


    frequency_scores = _map_unique(
        partial(_frequency_score_for, consensus=consensus),
        [c for c, alloy in zip(compositions, alloy_mask) if not alloy],
        n_jobs,
    )
    scores = np.array([frequency_scores[c] for c, alloy in zip(compositions, alloy_mask) if not alloy])
    if exclude_nonchargeable:
        # Only compositions with no admissible assignment score exactly 0.
        scores = scores[scores > 0.0]

    ref_comp_fps, ref_struct_fps, ref_densities, ref_num_elements = reference_fingerprints(
        reference, cache_dir, n_jobs
    )
    comp_fps = parallel_map(composition_fingerprint, [s.composition for s in structures], n_jobs)
    struct_fps = parallel_map_with_stall_timeout(structure_fingerprint, structures, n_jobs)
    n_no_fingerprint = sum(fp is None for fp in struct_fps)
    if n_no_fingerprint:
        logger.info(f"{n_no_fingerprint} structure(s) could not be fingerprinted.")

    flags = parallel_map(_validity_flags, structures, n_jobs)
    comp_valid = np.array([f[0] for f in flags], dtype=bool)
    charge_balanced = np.array([f[1] for f in flags], dtype=bool)
    struct_valid = np.array([f[2] for f in flags], dtype=bool)
    # As in DiffCSP's `Crystal`, a structure that cannot be fingerprinted is invalid.
    valid = comp_valid & struct_valid & np.array([fp is not None for fp in struct_fps], dtype=bool)
    n_unknown = sum(f[3] for f in flags)
    if n_unknown:
        logger.warning(f"{n_unknown} structure(s) contain elements unknown to SMACT; counted as invalid.")
    # Parse failures stay in cov_precision's denominator as unmatched structures.
    cov_recall, cov_precision = compute_coverage(
        struct_fps + [None] * n_parse_failures,
        comp_fps + [None] * n_parse_failures,
        ref_struct_fps,
        ref_comp_fps,
        struc_cutoff,
        comp_cutoff,
    )
    wdist_density, wdist_num_elems = _wasserstein_distances(
        structures, valid, ref_densities, ref_num_elements, wasserstein_n_samples, wasserstein_seed
    )

    reference_distribution = reference_oxidation_state_distribution(
        reference, consensus, cache_dir, n_jobs
    )
    generated_distribution = _marginalized_distribution(compositions, consensus, n_jobs)
    n_non_alloy = int((~alloy_mask).sum()) + n_parse_failures

    return {
        "n_structures": n_total,
        "n_parse_failures": n_parse_failures,
        "frac_alloy_or_single_element": float(alloy_mask.sum() / n_total),
        "comp_valid": float(comp_valid.sum() / n_total),
        "struct_valid": float(struct_valid.sum() / n_total),
        "valid": float(valid.sum() / n_total),
        "comp_valid_non_alloy_or_single_element": (
            float(charge_balanced[~alloy_mask].sum() / n_non_alloy) if n_non_alloy else float("nan")
        ),
        "frac_charge_neutral": _frac_charge_neutral(labels, alloy_mask, n_parse_failures),
        "avg_oxidation_state_frequency_score": (
            float(scores.mean()) if len(scores) else float("nan")
        ),
        "oxidation_state_distance": mean_js_distance(generated_distribution, reference_distribution),
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "wdist_density": wdist_density,
        "wdist_num_elems": wdist_num_elems,
    }
