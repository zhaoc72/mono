"""Visualization helpers for segmentation and depth outputs."""
from __future__ import annotations

import colorsys
from typing import Optional, Sequence

import cv2
import numpy as np

from models.segmenter import InstanceMask
from models.depth_estimator import DepthResult


def _color_palette(n: int) -> np.ndarray:
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for h in hues:
        rgb = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    return np.asarray(colors, dtype=np.uint8)


def draw_instance_masks(
    image: np.ndarray,
    masks: Sequence[InstanceMask],
    alpha: float = 0.5,
    draw_labels: bool = True,
    draw_scores: bool = True,
) -> np.ndarray:
    """Overlay instance masks with optional labels on an RGB image."""

    output = image.copy()
    overlay = image.copy()
    colors = _color_palette(max(len(masks), 1))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, instance in enumerate(masks):
        color = colors[idx % len(colors)].tolist()
        mask = instance.mask.astype(bool)
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        if draw_labels or draw_scores:
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            y = int(np.median(ys))
            x = int(np.median(xs))
            label_parts = []
            if draw_labels and instance.category:
                label_parts.append(str(instance.category))
            if draw_scores and instance.score is not None:
                label_parts.append(f"{instance.score:.2f}")
            if label_parts:
                cv2.putText(output, " | ".join(label_parts), (x, y), font, 0.5, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, dst=output)
    return output


def draw_depth_heatmap(depth: DepthResult, cmap: str = "plasma") -> np.ndarray:
    """Convert a depth result to a heatmap visualization."""

    return depth.to_colormap(cmap)


def stack_visualizations(
    image: np.ndarray,
    segmentation: Optional[np.ndarray],
    depth: Optional[np.ndarray],
) -> np.ndarray:
    """Stack available visualizations horizontally for quick inspection."""

    visuals = [image]
    if segmentation is not None:
        visuals.append(segmentation)
    if depth is not None:
        visuals.append(depth)
    visuals_resized = [cv2.resize(v, (image.shape[1], image.shape[0])) for v in visuals]
    return np.concatenate(visuals_resized, axis=1)
