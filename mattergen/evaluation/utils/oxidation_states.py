"""Oxidation-state commonality and marginalised oxidation-state distribution distance.

Scores how "chemically typical" a generated composition's oxidation states are against
SMACT's ICSD24 species-occurrence statistics, and compares the resulting oxidation-state
distribution against a reference dataset without committing to any single labelling.

Both metrics search the same admissible-assignment space: every combination of oxidation
states (one per distinct element in a composition) that is simultaneously charge-neutral
(``smact.neutral_ratios``) and Pauling-consistent (``smact.screening.pauling_test``), drawn
from each element's ICSD24-observed states. ``enumerate_admissible_assignments`` performs
this search once; ``oxidation_state_commonality`` and ``marginalized_occurrences`` are two
different reductions over its result -- the former keeps only the best assignment (the
geometric mean, across elements, of each element's ICSD occurrence proportion in that
assignment), the latter weights every admissible assignment by its ICSD-proportion product
and accumulates expected per-(element, oxidation state) occurrence across a dataset.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.spatial.distance import jensenshannon
from smact import _gcd_recursive, element_dictionary, neutral_ratios
from smact.screening import pauling_test
from smact.utils.oxidation import ICSD24OxStatesFilter

# Whether a composition is a pure-metal alloy or a single-species compound, for which ionic
# oxidation state is undefined or trivial. Already a `mattergen` dependency (`neutral-layer`),
# used throughout this fork -- reused here rather than reimplemented.
from neutral_layer.data.filtering import is_zero_record as is_alloy_or_single_element

__all__ = [
    "is_alloy_or_single_element",
    "enumerate_admissible_assignments",
    "oxidation_state_commonality",
    "marginalized_occurrences",
    "aggregate_marginalized_distribution",
    "mean_js_distance",
]


def _parse_species_ox_state(element: str, species: str) -> int:
    if species.endswith("+"):
        return int(species[len(element) : -1]) if len(species) > len(element) + 1 else 1
    if species.endswith("-"):
        return -int(species[len(element) : -1]) if len(species) > len(element) + 1 else -1
    return 0


@lru_cache(maxsize=None)
def _load_icsd_species_proportions(consensus: int) -> dict[str, dict[int, float]]:
    """Per-element ``{oxidation_state: ICSD occurrence proportion in [0, 1]}`` for species
    with at least `consensus` literature occurrences (smact's ICSD24 filter). `include_zero`
    is always False: OS=0 is not a meaningful "oxidation state commonality" candidate here.
    """
    df = ICSD24OxStatesFilter().get_species_occurrences_df(consensus=consensus, include_zero=False)
    proportions: dict[str, dict[int, float]] = {}
    for _, row in df.iterrows():
        ox_state = _parse_species_ox_state(row["element"], row["species"])
        proportions.setdefault(row["element"], {})[ox_state] = row["species_proportion (%)"] / 100.0
    return proportions


@lru_cache(maxsize=None)
def enumerate_admissible_assignments(
    elem_symbols: tuple[str, ...], counts: tuple[int, ...], consensus: int
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    """Every charge-neutral, Pauling-consistent oxidation-state assignment for a composition,
    each paired with a weight equal to the product, over elements, of that element's ICSD24
    occurrence proportion for the assigned state. `elem_symbols`/`counts` give a composition's
    distinct elements and their atom counts; every atom of a given element is assumed to share
    one oxidation state per assignment, since composition alone can't resolve mixed-valence
    sites. Empty if any element has no ICSD24-admissible states at this `consensus`, or no
    assignment is both charge-neutral and Pauling-consistent.
    """
    proportions = _load_icsd_species_proportions(consensus)
    if any(sym not in proportions for sym in elem_symbols):
        return ()

    gcd_val = _gcd_recursive(*counts)
    stoichs = [(int(c // gcd_val),) for c in counts]
    threshold = max(int(c // gcd_val) for c in counts)

    space = element_dictionary(elem_symbols)
    electronegs = [space[sym].pauling_eneg for sym in elem_symbols]
    ox_combos = [list(proportions[sym].keys()) for sym in elem_symbols]

    assignments: list[tuple[float, tuple[int, ...]]] = []
    for ox_states in itertools.product(*ox_combos):
        if not neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold):
            continue
        try:
            if not pauling_test(ox_states, electronegs):
                continue
        except TypeError:
            # Pauling test is inconclusive for this element combination (e.g. tied
            # electronegativities); treat the assignment as admissible rather than reject it.
            pass
        weight = 1.0
        for sym, ox_state in zip(elem_symbols, ox_states):
            weight *= proportions[sym][ox_state]
        if weight > 0:
            assignments.append((weight, ox_states))
    return tuple(assignments)


def oxidation_state_commonality(
    elem_symbols: tuple[str, ...], counts: tuple[int, ...], consensus: int
) -> float:
    """Geometric mean, across elements, of each element's ICSD24 occurrence proportion in the
    single best (highest-weight) admissible assignment for a composition -- how "typical" its
    most plausible oxidation states are. `0.0` if no admissible assignment exists.

    Callers are responsible for the alloy/single-element trivial case
    (`is_alloy_or_single_element`) themselves: whether such compositions should score `1.0`
    (only one possible assignment, OS=0) or be excluded entirely is a modelling choice about
    what "commonality" should measure, not a property of the assignment search itself.
    """
    assignments = enumerate_admissible_assignments(elem_symbols, counts, consensus)
    if not assignments:
        return 0.0
    best_weight = max(weight for weight, _ in assignments)
    return best_weight ** (1.0 / len(elem_symbols))


def marginalized_occurrences(
    elem_symbols: tuple[str, ...], counts: tuple[int, ...], consensus: int
) -> dict[str, dict[int, float]] | None:
    """Expected per-(element, oxidation state) *occurrence* for one composition, marginalising
    over every admissible assignment instead of committing to the best one (see module
    docstring). Weighting matches ICSD24's own per-entry, not per-atom, occurrence convention:
    a structure contributes total mass 1 to each of its elements, split fractionally across
    admissible states, regardless of how many sites/atoms realize it.

    Returns `None` for compositions with no admissible assignment (nothing to marginalise
    over). This is unconditional -- unlike the alloy/single-element and non-chargeable
    exclusions applied on top of `oxidation_state_commonality`, there is no meaningful
    marginal distribution to fall back to here.
    """
    assignments = enumerate_admissible_assignments(elem_symbols, counts, consensus)
    if not assignments:
        return None

    total_weight = sum(weight for weight, _ in assignments)
    expected: dict[str, Counter] = defaultdict(Counter)
    for weight, ox_states in assignments:
        q = weight / total_weight
        for sym, ox_state in zip(elem_symbols, ox_states):
            expected[sym][ox_state] += q
    return {sym: dict(counter) for sym, counter in expected.items()}


def aggregate_marginalized_distribution(
    per_structure: Iterable[dict[str, dict[int, float]] | None]
) -> dict[str, dict[int, float]]:
    """Pool per-structure expected occurrences (see `marginalized_occurrences`) into the
    dataset-level marginalised distribution p_D(z|a), normalized per element. Structures with
    `None` (alloy/single-element or non-chargeable compositions) contribute nothing.
    """
    totals: dict[str, Counter] = defaultdict(Counter)
    for occurrences in per_structure:
        if occurrences is None:
            continue
        for sym, dist in occurrences.items():
            for ox_state, expected_occurrence in dist.items():
                totals[sym][ox_state] += expected_occurrence

    distribution: dict[str, dict[int, float]] = {}
    for sym, counter in totals.items():
        total = sum(counter.values())
        if total > 0:
            distribution[sym] = {ox_state: n / total for ox_state, n in counter.items()}
    return distribution


def _js_distance(p: dict[int, float], q: dict[int, float]) -> float:
    """Jensen-Shannon distance (the metric, i.e. sqrt of the divergence) between two
    per-oxidation-state distributions for one element."""
    support = sorted(set(p) | set(q))
    p_vec = np.array([p.get(s, 0.0) for s in support])
    q_vec = np.array([q.get(s, 0.0) for s in support])
    return float(jensenshannon(p_vec, q_vec, base=2))


def mean_js_distance(
    dist_a: dict[str, dict[int, float]], dist_b: dict[str, dict[int, float]]
) -> float:
    """Unweighted average per-element JS distance between two per-element oxidation-state
    distributions, over elements present in both. `nan` if there is no overlap.
    """
    per_element = [_js_distance(dist, dist_b[el]) for el, dist in dist_a.items() if el in dist_b]
    return sum(per_element) / len(per_element) if per_element else float("nan")
