"""Run-declustered extreme episode extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dyn_evt_pdm.types import EventInterval


@dataclass(frozen=True, slots=True)
class ExtremeCluster:
    """One cluster of exceedances separated by short non-extreme gaps."""

    interval: EventInterval
    exceedance_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        """Number of exceedances, not calendar duration."""

        return len(self.exceedance_indices)


def extract_clusters(exceedances: np.ndarray, *, run_length: int) -> list[ExtremeCluster]:
    """Group exceedances when consecutive exceedances are at most ``run_length`` apart.

    ``run_length`` is the maximum number of non-exceedance samples allowed between
    two exceedances in the same cluster. A value of zero therefore creates clusters
    from directly consecutive exceedances only.
    """

    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    flags = np.asarray(exceedances, dtype=bool)
    indices = np.flatnonzero(flags)
    if len(indices) == 0:
        return []

    grouped: list[list[int]] = [[int(indices[0])]]
    for previous, current in zip(indices[:-1], indices[1:], strict=True):
        gap = int(current - previous - 1)
        if gap <= run_length:
            grouped[-1].append(int(current))
        else:
            grouped.append([int(current)])

    return [
        ExtremeCluster(
            interval=EventInterval(start=group[0], end=group[-1], label="extreme_cluster"),
            exceedance_indices=tuple(group),
        )
        for group in grouped
    ]


def cluster_maxima(values: np.ndarray, clusters: list[ExtremeCluster]) -> np.ndarray:
    """Return the maximum value observed in each cluster."""

    array = np.asarray(values, dtype=float)
    maxima = [float(np.max(array[list(cluster.exceedance_indices)])) for cluster in clusters]
    return np.asarray(maxima, dtype=np.float64)
