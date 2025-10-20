"""Visualization helpers for saving segmentation and depth predictions."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


COLORMAP = cv2.COLORMAP_MAGMA


def overlay_segmentation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an overlay of the segmentation mask on the image."""
    mask_rgb = np.zeros_like(image)
    mask_rgb[:, :, 1] = (mask * 255).astype(np.uint8)
    return cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)


def save_segmentation_mask(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """Overlay the segmentation mask on the image and save it."""
    overlay = overlay_segmentation(image, mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Return a depth visualization using a perceptual colormap."""
    depth_normalized = depth - depth.min()
    max_val = depth_normalized.max()
    if max_val > 0:
        depth_normalized = depth_normalized / max_val
    depth_img = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_img, COLORMAP)


def save_depth_map(depth: np.ndarray, path: Path) -> None:
    """Apply a colormap to the depth map and save it."""
    colored = colorize_depth(depth)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), colored)
