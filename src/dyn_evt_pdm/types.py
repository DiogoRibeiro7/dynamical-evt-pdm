"""Shared type aliases and small validated domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
RegimeName = NewType("RegimeName", str)


@dataclass(frozen=True, slots=True)
class EventInterval:
    """Inclusive integer interval representing one physical or alarm event."""

    start: int
    end: int
    label: str = "event"

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("event start must be non-negative")
        if self.end < self.start:
            raise ValueError("event end must not precede start")

    @property
    def duration(self) -> int:
        """Return interval duration in samples, including both endpoints."""

        return self.end - self.start + 1

    def overlaps(self, other: "EventInterval") -> bool:
        """Return whether this interval overlaps another interval."""

        return self.start <= other.end and other.start <= self.end
