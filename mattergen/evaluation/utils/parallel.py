"""Process-pool helpers for the per-structure evaluation metrics."""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Callable, Sequence

from mattergen.evaluation.utils.logging import logger


def default_n_jobs() -> int:
    """Number of CPUs allocated to this process.

    Uses the scheduler's allocation (Slurm ``SLURM_CPUS_PER_TASK``, PBS ``NCPUS``)
    when set. Otherwise uses the CPUs this process may run on, which, unlike
    ``os.cpu_count()``, respects the cpuset of a shared node.
    """
    for var in ("SLURM_CPUS_PER_TASK", "NCPUS"):
        raw = os.environ.get(var)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                logger.warning(f"Ignoring non-integer {var}={raw!r}")
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def parallel_map(
    func: Callable[[Any], Any],
    items: Sequence,
    n_jobs: int | None = None,
    min_items_for_parallelism: int = 32,
) -> list:
    """``[func(item) for item in items]``, in a process pool for large inputs.

    ``func`` must be picklable (a top-level function or a ``functools.partial``
    of one). Worker exceptions propagate to the caller.
    """
    items = list(items)
    if not items:
        return []
    n_jobs = n_jobs if n_jobs is not None else default_n_jobs()
    if n_jobs <= 1 or len(items) < min_items_for_parallelism:
        return [func(item) for item in items]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(func, items))


def parallel_map_with_stall_timeout(
    func: Callable[[Any], Any],
    items: Sequence,
    n_jobs: int | None = None,
    stall_timeout: float = 120.0,
) -> list:
    """``parallel_map``, returning ``None`` for items whose worker never returns.

    A few generated structures (about 1 in 850 of FlowMM's samples) send
    CrystalNN's Voronoi tessellation into an effectively non-terminating Qhull
    call, which Python signal handlers cannot interrupt, so the worker has to be
    abandoned. Once no task has completed for ``stall_timeout`` seconds, the
    remaining items are returned as ``None`` and their workers killed. Such
    structures should not be filtered out in advance: nearly all of them still
    fall within the coverage cutoffs.
    """
    items = list(items)
    if not items:
        return []
    n_jobs = n_jobs if n_jobs is not None else default_n_jobs()
    if n_jobs <= 1:
        return [func(item) for item in items]

    results: list = [None] * len(items)
    pool = ProcessPoolExecutor(max_workers=n_jobs)
    try:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        pending = set(futures)
        last_progress = time.monotonic()
        while pending:
            done, pending = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)
            if done:
                last_progress = time.monotonic()
                for future in done:
                    results[futures[future]] = future.result()
            elif time.monotonic() - last_progress > stall_timeout:
                logger.warning(
                    f"No task completed in {stall_timeout:.0f} s; "
                    f"abandoning {len(pending)} stuck item(s) as None."
                )
                break
    finally:
        # shutdown() cannot stop a running task, so stuck workers are killed
        # directly through the executor's private process table.
        for process in list(getattr(pool, "_processes", {}).values()):
            if process.is_alive():
                process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    return results
