"""Oxidation-state frequency score and marginalised oxidation-state distribution distance.

Scores how chemically typical a composition's oxidation states are under SMACT's ICSD24
species-occurrence statistics, and compares dataset-level oxidation-state distributions without
committing to a single labelling per composition.

Both metrics use the same admissible assignments: one oxidation state per distinct element, drawn
from its ICSD24-observed states, such that the composition is charge-neutral
(``smact.neutral_ratios``) and Pauling-consistent (``smact.screening.pauling_test``). An
assignment's weight is the product over elements of their ICSD24 occurrence proportions.
``oxidation_state_frequency_score`` keeps the best assignment, while ``marginalized_occurrences``
averages over all of them in proportion to weight.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterable

import numpy as np
from pymatgen.core import Element, Structure
from scipy.spatial.distance import jensenshannon
from smact import _gcd_recursive, element_dictionary, neutral_ratios
from smact.screening import pauling_test
from smact.utils.oxidation import ICSD24OxStatesFilter

# True for pure-metal alloys and single-element compositions, where ionic oxidation states are
# undefined or trivial.
from neutral_layer.data.filtering import is_zero_record as is_alloy_or_single_element

__all__ = [
    "composition_symbols_and_counts",
    "is_alloy_or_single_element",
    "enumerate_admissible_assignments",
    "oxidation_state_frequency_score",
    "marginalized_occurrences",
    "aggregate_marginalized_distribution",
    "mean_js_distance",
]


def composition_symbols_and_counts(structure: Structure) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Distinct element symbols and atom counts, sorted by atomic number.

    The canonical order maximises cache hits in the memoized
    `enumerate_admissible_assignments`.
    """
    elem_counter = Counter(structure.atomic_numbers)
    ordered_z = sorted(elem_counter)
    symbols = tuple(str(Element.from_Z(z)) for z in ordered_z)
    counts = tuple(int(elem_counter[z]) for z in ordered_z)
    return symbols, counts


def _parse_species_ox_state(element: str, species: str) -> int:
    if species.endswith("+"):
        return int(species[len(element) : -1]) if len(species) > len(element) + 1 else 1
    if species.endswith("-"):
        return -int(species[len(element) : -1]) if len(species) > len(element) + 1 else -1
    return 0


@lru_cache(maxsize=None)
def _load_icsd_species_proportions(consensus: int) -> dict[str, dict[int, float]]:
    """Per-element ``{oxidation_state: ICSD24 occurrence proportion in [0, 1]}`` for species
    with at least `consensus` occurrences. Oxidation state 0 is excluded, since it is not a
    meaningful frequency-score candidate.
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
    """All admissible `(weight, ox_states)` pairs for a composition (see module docstring).

    `elem_symbols` and `counts` are the distinct elements and their atom counts. All atoms of an
    element share one oxidation state, since composition alone cannot resolve mixed valence.
    Empty if any element has no ICSD24 states at this `consensus`, or no assignment is
    admissible.
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


def oxidation_state_frequency_score(
    elem_symbols: tuple[str, ...], counts: tuple[int, ...], consensus: int
) -> float:
    """Geometric mean, across elements, of the ICSD24 occurrence proportions in the
    highest-weight admissible assignment. `0.0` if no assignment is admissible.

    Alloys and single-element compositions (`is_alloy_or_single_element`) are not special-cased:
    whether they score 1.0 or are excluded is left to the caller.
    """
    assignments = enumerate_admissible_assignments(elem_symbols, counts, consensus)
    if not assignments:
        return 0.0
    best_weight = max(weight for weight, _ in assignments)
    return best_weight ** (1.0 / len(elem_symbols))


def marginalized_occurrences(
    elem_symbols: tuple[str, ...], counts: tuple[int, ...], consensus: int
) -> dict[str, dict[int, float]] | None:
    """Expected per-(element, oxidation state) occurrence for one composition, marginalising over
    admissible assignments with probability proportional to weight.

    Counts are per entry, not per atom, matching ICSD24's occurrence convention: each element
    receives total mass 1, split across its states, however many atoms it has. Returns `None` if
    no assignment is admissible.
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
    dataset-level distribution p_D(z|a), normalized per element. `None` entries contribute
    nothing.
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
    """Base-2 Jensen-Shannon distance (the square root of the divergence)."""
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
