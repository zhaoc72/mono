"""Geometry helper utilities."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: Dict[str, float],
    mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a depth map to 3D points and return (points, pixels)."""

    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    if mask is not None:
        valid = mask.astype(bool)
    else:
        valid = depth > 0

    xs = xs[valid]
    ys = ys[valid]
    depth_vals = depth[valid]

    if max_points is not None and xs.size > max_points:
        idx = np.random.choice(xs.size, max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
        depth_vals = depth_vals[idx]

    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")

    x = (xs - cx) * depth_vals / fx
    y = (ys - cy) * depth_vals / fy
    z = depth_vals
    points = np.stack([x, y, z], axis=1)
    pixels = np.stack([xs, ys], axis=1)
    return points.astype(np.float32), pixels.astype(np.int32)
