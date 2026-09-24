"""Tests for `mattergen.evaluation.evaluate_os` and its parallel helpers.

All fixtures are small synthetic structures and a miniature MP-20-OS-shaped split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from scipy.stats import wasserstein_distance

import mattergen.evaluation.evaluate_os as eos
from mattergen.evaluation.reference.mp20_os import load_mp20_os_reference_dataset
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils import parallel
from mattergen.evaluation.utils.oxidation_states import (
    aggregate_marginalized_distribution,
    composition_symbols_and_counts,
    is_alloy_or_single_element,
    marginalized_occurrences,
    mean_js_distance,
    oxidation_state_frequency_score,
)

_LATTICE = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]
_COORDS = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.4, 0.6, 0.2], [0.0, 0.0, 0.75]]


def _structure(species: list[str]) -> Structure:
    return Structure(lattice=_LATTICE, species=species, coords=_COORDS)


def _sample_structures() -> list[Structure]:
    return [
        _structure(["Na", "Na", "Cl", "Cl"]),
        _structure(["Fe", "Fe", "Ni", "Mn"]),  # alloy
        _structure(["Ti", "Ti", "O", "O"]),
        _structure(["Na", "Na", "Cl", "Cl"]),  # repeated composition
    ]


def _labelled_structures() -> list[Structure]:
    return [
        _structure(["Na+", "Na+", "Cl-", "Cl-"]),  # neutral
        _structure(["Fe", "Fe", "Ni", "Mn"]),  # alloy, excluded
        _structure(["Ti4+", "Ti4+", "O2-", "O2-"]),  # net charge +4
    ]


def _write_mp20_os_split(root: Path, split: str, structures: list[Structure]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "atomic_numbers.npy", np.concatenate([s.atomic_numbers for s in structures]))
    np.save(split_dir / "cell.npy", np.array([s.lattice.matrix for s in structures]))
    np.save(split_dir / "pos.npy", np.concatenate([s.frac_coords for s in structures]))
    np.save(split_dir / "num_atoms.npy", np.array([len(s) for s in structures]))


@pytest.fixture
def reference_dataset(tmp_path: Path) -> ReferenceDataset:
    root = tmp_path / "mp-20-os"
    _write_mp20_os_split(root, "test", _sample_structures())
    return load_mp20_os_reference_dataset(root, "test")


# -----------------------------#
# Parallel helpers
# -----------------------------#


def _square(x: int) -> int:
    return x * x


def _raise(x: int) -> int:
    raise ValueError(f"boom {x}")


@pytest.mark.parametrize("n_jobs,min_items", [(1, 32), (2, 0)])
def test_parallel_map_matches_serial(n_jobs, min_items):
    items = list(range(10))
    result = parallel.parallel_map(_square, items, n_jobs=n_jobs, min_items_for_parallelism=min_items)
    assert result == [_square(i) for i in items]


def test_parallel_map_empty_items():
    assert parallel.parallel_map(_square, [], n_jobs=2) == []


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallel_map_propagates_worker_exceptions(n_jobs):
    with pytest.raises(ValueError, match="boom"):
        parallel.parallel_map(_raise, [1, 2, 3], n_jobs=n_jobs, min_items_for_parallelism=0)


def test_parallel_map_with_stall_timeout_matches_serial():
    items = list(range(10))
    assert parallel.parallel_map_with_stall_timeout(_square, items, n_jobs=2) == [
        _square(i) for i in items
    ]


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@pytest.mark.parametrize("var", ["SLURM_CPUS_PER_TASK", "NCPUS"])
def test_default_n_jobs_uses_scheduler_allocation(monkeypatch, var):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("NCPUS", raising=False)
    monkeypatch.setenv(var, "7")
    assert parallel.default_n_jobs() == 7


def test_default_n_jobs_falls_back_to_available_cpus(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-an-int")
    monkeypatch.delenv("NCPUS", raising=False)
    assert parallel.default_n_jobs() == _available_cpus()


# -----------------------------#
# CIF loading
# -----------------------------#


def test_load_cif_structures_counts_failures_and_keeps_order_and_oxidation_states(tmp_path):
    zip_path = tmp_path / "generated_crystals_cif.zip"
    structures = _labelled_structures()
    with ZipFile(zip_path, "w") as zf:
        # Written out of order, to check the numeric gen_{i} ordering.
        for i in (2, 0, 1):
            zf.writestr(f"gen_{i}.cif", str(CifWriter(structures[i])))
        zf.writestr("gen_10.cif", "this is not a CIF")

    loaded, n_failed = eos.load_cif_structures(zip_path, n_jobs=1)

    assert n_failed == 1
    assert [s.composition.reduced_formula for s in loaded] == [
        s.composition.reduced_formula for s in structures
    ]
    assert sorted(eos._oxidation_state_labels(loaded[0])) == [-1, -1, 1, 1]


def test_load_cif_structures_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        eos.load_cif_structures(tmp_path)


# -----------------------------#
# Reference caches
# -----------------------------#


def test_reference_fingerprint_cache_hit_avoids_recompute(tmp_path, reference_dataset, monkeypatch):
    cache_dir = tmp_path / "cache"
    comp_fps, struct_fps, densities, num_elements = eos.reference_fingerprints(
        reference_dataset, cache_dir, n_jobs=1
    )

    def _explode(*args, **kwargs):
        raise AssertionError("should not recompute on a cache hit")

    monkeypatch.setattr(eos, "composition_fingerprint", _explode)
    monkeypatch.setattr(eos, "structure_fingerprint", _explode)
    comp_fps2, struct_fps2, densities2, num_elements2 = eos.reference_fingerprints(
        reference_dataset, cache_dir, n_jobs=1
    )

    for a, b in zip(comp_fps, comp_fps2):
        np.testing.assert_allclose(a, b)
    for a, b in zip(struct_fps, struct_fps2):
        if a is None:
            assert b is None
        else:
            np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(densities, densities2)
    np.testing.assert_array_equal(num_elements, num_elements2)


def test_reference_fingerprint_cache_recomputes_when_stale(tmp_path, reference_dataset):
    cache_dir = tmp_path / "cache"
    eos.reference_fingerprints(reference_dataset, cache_dir, n_jobs=1)
    meta_path = cache_dir / "reference_fingerprints_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["n_entries"] += 1
    meta_path.write_text(json.dumps(meta))

    comp_fps, _, _, _ = eos.reference_fingerprints(reference_dataset, cache_dir, n_jobs=1)

    assert len(comp_fps) == len(reference_dataset)
    assert json.loads(meta_path.read_text())["n_entries"] == len(reference_dataset)


def test_oxidation_state_distribution_cache_round_trips_int_keys(tmp_path, reference_dataset, monkeypatch):
    cache_dir = tmp_path / "cache"
    dist = eos.reference_oxidation_state_distribution(reference_dataset, 3, cache_dir, n_jobs=1)
    assert dist

    def _explode(*args, **kwargs):
        raise AssertionError("should not recompute on a cache hit")

    monkeypatch.setattr(eos, "marginalized_occurrences", _explode)
    dist2 = eos.reference_oxidation_state_distribution(reference_dataset, 3, cache_dir, n_jobs=1)

    assert dist2 == dist
    assert all(isinstance(k, int) for per_state in dist2.values() for k in per_state)


# -----------------------------#
# Metrics
# -----------------------------#


def test_evaluate_os_returns_all_metrics(reference_dataset):
    metrics = eos.evaluate_os(_sample_structures(), reference_dataset, n_jobs=1)
    assert set(metrics) == {
        "n_structures", "n_parse_failures", "frac_alloy_or_single_element",
        "comp_valid", "struct_valid", "valid", "comp_valid_non_alloy_or_single_element", "frac_charge_neutral",
        "avg_oxidation_state_frequency_score", "oxidation_state_distance",
        "cov_recall", "cov_precision", "wdist_density", "wdist_num_elems",
    }
    assert metrics["frac_alloy_or_single_element"] == pytest.approx(0.25)
    assert np.isnan(metrics["frac_charge_neutral"])  # no oxidation-state labels


def test_frequency_score_matches_direct_computation(reference_dataset):
    structures = _sample_structures()
    metrics = eos.evaluate_os(structures, reference_dataset, n_jobs=1)

    compositions = [composition_symbols_and_counts(s) for s in structures]
    scores = [
        oxidation_state_frequency_score(symbols, counts, 3)
        for symbols, counts in compositions
        if not is_alloy_or_single_element(symbols)
    ]
    expected = np.mean([score for score in scores if score > 0])
    assert metrics["avg_oxidation_state_frequency_score"] == pytest.approx(expected)


def test_oxidation_state_distance_matches_direct_computation(reference_dataset):
    structures = _sample_structures()
    metrics = eos.evaluate_os(structures, reference_dataset, n_jobs=1)

    def distribution(strucs):
        compositions = [composition_symbols_and_counts(s) for s in strucs]
        return aggregate_marginalized_distribution(
            None if is_alloy_or_single_element(s) else marginalized_occurrences(s, c, 3)
            for s, c in compositions
        )

    reference_structures = [entry.structure for entry in reference_dataset]
    expected = mean_js_distance(distribution(structures), distribution(reference_structures))
    assert metrics["oxidation_state_distance"] == pytest.approx(expected, nan_ok=True)


def test_charge_neutrality_excludes_alloys_and_counts_parse_failures(reference_dataset):
    structures = _labelled_structures()
    no_failures = eos.evaluate_os(structures, reference_dataset, n_jobs=1)
    one_failure = eos.evaluate_os(structures, reference_dataset, n_parse_failures=1, n_jobs=1)

    # NaCl is neutral, Ti2O2 is not and the alloy is excluded.
    assert no_failures["frac_charge_neutral"] == pytest.approx(1 / 2)
    assert one_failure["frac_charge_neutral"] == pytest.approx(1 / 3)


def test_parse_failures_stay_in_denominators(reference_dataset):
    structures = _sample_structures()
    base = eos.evaluate_os(structures, reference_dataset, n_jobs=1)
    failed = eos.evaluate_os(structures, reference_dataset, n_parse_failures=2, n_jobs=1)

    n, n_non_alloy = len(structures), 3
    assert failed["n_structures"] == n + 2
    for key in ("comp_valid", "struct_valid", "valid", "cov_precision", "frac_alloy_or_single_element"):
        assert failed[key] == pytest.approx(base[key] * n / (n + 2))
    assert failed["comp_valid_non_alloy_or_single_element"] == pytest.approx(
        base["comp_valid_non_alloy_or_single_element"] * n_non_alloy / (n_non_alloy + 2)
    )
    # Averages and distributions cover parsed structures only.
    for key in ("avg_oxidation_state_frequency_score", "oxidation_state_distance", "cov_recall"):
        assert failed[key] == pytest.approx(base[key], nan_ok=True)


def test_wasserstein_distances_use_valid_structures_only():
    structures = _sample_structures()
    valid = np.array([True, False, True, False])
    ref_densities = np.array([1.0, 2.0, 3.0])
    ref_num_elements = np.array([2, 2, 3])

    wdist_density, wdist_num_elems = eos._wasserstein_distances(
        structures, valid, ref_densities, ref_num_elements, n_samples=0, seed=0
    )

    kept = [structures[0], structures[2]]
    assert wdist_density == pytest.approx(
        wasserstein_distance([s.density for s in kept], ref_densities)
    )
    assert wdist_num_elems == pytest.approx(
        wasserstein_distance([len(set(s.species)) for s in kept], ref_num_elements)
    )


def test_evaluate_os_serial_and_parallel_agree(reference_dataset):
    structures = _sample_structures()
    serial = eos.evaluate_os(structures, reference_dataset, n_jobs=1)
    parallel_metrics = eos.evaluate_os(structures, reference_dataset, n_jobs=2)
    for key in serial:
        assert serial[key] == pytest.approx(parallel_metrics[key], nan_ok=True)
