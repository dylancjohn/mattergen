# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CDVAE-style structural/compositional diversity metrics: coverage (COV-Recall/COV-Precision) and
Wasserstein distance between generated and reference distributions of density and N-ary (number of
distinct elements). These are the metrics DiffCSP, CrystalFlow and FlowMM all report alongside their
own paper-specific numbers -- see `mattergen.evaluation.utils.fingerprints` for provenance and the
exact fingerprint/coverage definitions reproduced here.

`wdist_prop` (Wasserstein distance on a pretrained property-predictor's output) is intentionally not
included: it needs a per-dataset pretrained CDVAE-family regressor checkpoint this pipeline has no
equivalent of.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing
from pandas import DataFrame
from scipy.stats import wasserstein_distance

from mattergen.evaluation.metrics.core import BaseMetric, BaseMetricsCapability
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.fingerprints import (
    compute_coverage,
    composition_fingerprint,
    structure_fingerprint,
)
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary

# In-process memoized (comp, struct) reference-dataset fingerprints, keyed by reference_dataset.name.
# Not a persistent cache (nothing is written to disk, and it's lost between process invocations) --
# purely to avoid redoing e.g. mp-20-os test's ~8,500 structures' worth of CrystalNN fingerprinting for
# every run evaluated against it within one process (a notebook looping over many runs, in particular).
_reference_fingerprints_cache: dict[str, tuple[list[np.ndarray], list[np.ndarray | None]]] = {}


def _reference_fingerprints(
    reference_dataset: ReferenceDataset,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    if reference_dataset.name not in _reference_fingerprints_cache:
        comp_fps = [composition_fingerprint(e.structure.composition) for e in reference_dataset]
        struct_fps = [structure_fingerprint(e.structure) for e in reference_dataset]
        _reference_fingerprints_cache[reference_dataset.name] = (comp_fps, struct_fps)
    return _reference_fingerprints_cache[reference_dataset.name]


class DiversityMetricsCapability(BaseMetricsCapability):
    name: str = "diversity_capability"

    """Capability for CDVAE-style diversity metrics. Independent of `StructureMetricsCapability`
    (mirrors `EnergyMetricsCapability`'s shape): built directly from `structure_summaries` and a
    `reference_dataset`, since coverage/Wasserstein need no relaxation, energies or structure-matcher
    novelty machinery.
    """

    def __init__(
        self,
        structure_summaries: list[MetricsStructureSummary],
        reference_dataset: ReferenceDataset,
        struc_cutoff: float = 0.4,
        comp_cutoff: float = 10.0,
        wasserstein_n_samples: int = 1000,
        wasserstein_seed: int = 0,
        n_failed_jobs: int = 0,
    ) -> None:
        super().__init__(structure_summaries=structure_summaries, n_failed_jobs=n_failed_jobs)
        self.structures = [s.structure for s in structure_summaries]
        self.reference_dataset = reference_dataset
        self.struc_cutoff = struc_cutoff
        self.comp_cutoff = comp_cutoff
        self.wasserstein_n_samples = wasserstein_n_samples
        self.wasserstein_seed = wasserstein_seed

    @cached_property
    def comp_fingerprints(self) -> list[np.ndarray]:
        return [composition_fingerprint(s.composition) for s in self.structures]

    @cached_property
    def struct_fingerprints(self) -> list[np.ndarray | None]:
        return [structure_fingerprint(s) for s in self.structures]

    @cached_property
    def reference_comp_fingerprints(self) -> list[np.ndarray]:
        return _reference_fingerprints(self.reference_dataset)[0]

    @cached_property
    def reference_struct_fingerprints(self) -> list[np.ndarray | None]:
        return _reference_fingerprints(self.reference_dataset)[1]

    @cached_property
    def reference_densities(self) -> numpy.typing.NDArray[np.float64]:
        return np.array([e.structure.density for e in self.reference_dataset])

    @cached_property
    def reference_num_elements(self) -> numpy.typing.NDArray[np.int_]:
        return np.array([len(set(e.structure.species)) for e in self.reference_dataset])

    @cached_property
    def coverage(self) -> dict[str, float]:
        """{"cov_recall": ..., "cov_precision": ...} -- see `fingerprints.compute_coverage`."""
        cov_recall, cov_precision = compute_coverage(
            self.struct_fingerprints,
            self.comp_fingerprints,
            self.reference_struct_fingerprints,
            self.reference_comp_fingerprints,
            self.struc_cutoff,
            self.comp_cutoff,
        )
        return {"cov_recall": cov_recall, "cov_precision": cov_precision}

    @cached_property
    def wasserstein_distances(self) -> dict[str, float]:
        """{"wdist_density": ..., "wdist_num_elems": ...}, computed on up to
        `wasserstein_n_samples` randomly-sampled generated structures with a valid structure
        fingerprint (matching upstream's protocol of sampling *valid* generated crystals before
        computing these two distances), against the full reference set. If fewer than
        `wasserstein_n_samples` are available, all of them are used with a warning instead of
        raising -- unlike upstream, which fails outright; this run's *n* then differs from the
        n=1000 DiffCSP/FlowMM/CrystalFlow protocol, which is worth knowing about but shouldn't
        crash an otherwise-complete evaluation.
        """
        valid_indices = [i for i, fp in enumerate(self.struct_fingerprints) if fp is not None]
        if len(valid_indices) < self.wasserstein_n_samples:
            logger.warning(
                f"Only {len(valid_indices)} generated structures have a valid structure "
                f"fingerprint, fewer than wasserstein_n_samples={self.wasserstein_n_samples}; "
                "using all of them instead of raising (n differs from the DiffCSP/FlowMM/"
                "CrystalFlow n=1000 protocol for this run)."
            )
            sampled_indices = valid_indices
        else:
            rng = np.random.RandomState(self.wasserstein_seed)
            sampled_indices = rng.choice(valid_indices, self.wasserstein_n_samples, replace=False)

        pred_densities = [self.structures[i].density for i in sampled_indices]
        pred_num_elements = [len(set(self.structures[i].species)) for i in sampled_indices]

        return {
            "wdist_density": wasserstein_distance(pred_densities, self.reference_densities),
            "wdist_num_elems": wasserstein_distance(pred_num_elements, self.reference_num_elements),
        }

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            data={
                "has_structure_fingerprint": [fp is not None for fp in self.struct_fingerprints],
            },
            index=[e.entry_id for e in self.dataset],
        )


# -----------------------------#
# Metrics
# -----------------------------#


@dataclass(frozen=True)
class BaseDiversityMetric(BaseMetric):
    required_capabilities = (DiversityMetricsCapability,)

    @property
    def name(self) -> str:
        return "base_diversity_metric"

    def __init__(self, diversity_capability: DiversityMetricsCapability, **kwargs):
        self.capability = diversity_capability


class CoverageRecall(BaseDiversityMetric):
    name = "cov_recall"

    @property
    def description(self) -> str:
        return (
            "COV-Recall (CDVAE/DiffCSP/FlowMM): fraction of the reference dataset's structures "
            "that have a matching generated structure (composition and structure fingerprint "
            "both within cutoff)."
        )

    @cached_property
    def value(self) -> float:
        return self.capability.coverage["cov_recall"]


class CoveragePrecision(BaseDiversityMetric):
    name = "cov_precision"

    @property
    def description(self) -> str:
        return (
            "COV-Precision (CDVAE/DiffCSP/FlowMM): fraction of generated structures that have a "
            "matching reference structure (composition and structure fingerprint both within "
            "cutoff)."
        )

    @cached_property
    def value(self) -> float:
        return self.capability.coverage["cov_precision"]


class DensityWassersteinDistance(BaseDiversityMetric):
    name = "wdist_density"

    @property
    def description(self) -> str:
        return (
            "Wasserstein distance (CDVAE/DiffCSP/FlowMM's wdist_density) between the density "
            "distributions of sampled and reference structures."
        )

    @cached_property
    def value(self) -> float:
        return self.capability.wasserstein_distances["wdist_density"]


class NumElementsWassersteinDistance(BaseDiversityMetric):
    name = "wdist_num_elems"

    @property
    def description(self) -> str:
        return (
            "Wasserstein distance (CDVAE/DiffCSP/FlowMM's wdist_num_elems) between the "
            "number-of-distinct-elements (N-ary) distributions of sampled and reference structures."
        )

    @cached_property
    def value(self) -> float:
        return self.capability.wasserstein_distances["wdist_num_elems"]
