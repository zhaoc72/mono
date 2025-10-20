"""Profiling helpers for tracking throughput."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AverageMeter:
    """Track running average of numeric values."""

    name: str
    val: float = 0.0
    sum: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
