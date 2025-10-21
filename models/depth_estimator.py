"""Depth Anything v2 estimator wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install transformers to use the depth estimator.") from exc


@dataclass
class DepthResult:
    depth: np.ndarray
    normalized: bool

    def to_colormap(self, cmap: str = "plasma") -> np.ndarray:
        import matplotlib.cm as cm

        mapper = cm.get_cmap(cmap)
        colored = mapper(self.depth / (self.depth.max() + 1e-6))
        return (colored[:, :, :3] * 255).astype(np.uint8)


class DepthAnythingEstimator:
    """Thin wrapper around the Hugging Face depth-estimation pipeline."""

    def __init__(
        self,
        model_id: str,
        device: Optional[int] = None,
        normalize_depth: bool = True,
    ) -> None:
        self.pipeline = pipeline("depth-estimation", model=model_id, device=device)
        self.normalize_depth = bool(normalize_depth)

    def estimate(self, image: Image.Image) -> DepthResult:
        output = self.pipeline(image)
        depth = output["depth"]
        if hasattr(depth, "numpy"):
            depth_np = depth.numpy()
        else:
            depth_np = np.array(depth)
        depth_np = depth_np.astype("float32")
        if self.normalize_depth:
            depth_min = float(depth_np.min())
            depth_max = float(depth_np.max())
            if depth_max > depth_min:
                depth_np = (depth_np - depth_min) / (depth_max - depth_min)
        return DepthResult(depth=depth_np, normalized=self.normalize_depth)


__all__ = ["DepthAnythingEstimator", "DepthResult"]
