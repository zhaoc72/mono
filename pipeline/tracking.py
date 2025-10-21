"""Utility classes for maintaining stable IDs across video frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Track:
    instance_id: int
    category: Optional[str]
    score: Optional[float]
    last_mask: np.ndarray


class TrackManager:
    """A minimalistic track manager using IoU based association."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def assign(self, masks: List[np.ndarray], categories: List[Optional[str]], scores: List[Optional[float]]) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        for mask, category, score in zip(masks, categories, scores):
            best_iou = 0.0
            best_track_id = None
            for track_id, track in self.tracks.items():
                iou_val = self._iou(mask, track.last_mask)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track_id = track_id
            if best_track_id is not None and best_iou >= self.iou_threshold:
                assignments[id(mask)] = best_track_id
                self.tracks[best_track_id].last_mask = mask
                self.tracks[best_track_id].score = score
            else:
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = Track(
                    instance_id=track_id,
                    category=category,
                    score=score,
                    last_mask=mask,
                )
                assignments[id(mask)] = track_id
        return assignments
