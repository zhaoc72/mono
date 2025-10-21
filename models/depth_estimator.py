"""Depth Anything v2 estimator wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install transformers to use the depth estimator.") from exc

try:  # pragma: no cover - optional dependency
    import torch
    from depth_anything_v2.depth_anything import DepthAnythingV2
    from torchvision import transforms
except ImportError:  # pragma: no cover - optional dependency
    DepthAnythingV2 = None
    torch = None
    transforms = None


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
        checkpoint: Optional[str] = None,
        encoder: Optional[str] = None,
    ) -> None:
        self.normalize_depth = bool(normalize_depth)
        self._pipeline = None
        self._local_model = None
        self._device = device
        self._transform = None

        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path and checkpoint_path.is_file():
            if DepthAnythingV2 is None or torch is None or transforms is None:
                raise ImportError(
                    "depth_anything_v2 and torchvision are required for local checkpoints."
                )
            device_str = f"cuda:{device}" if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
            self._local_model = DepthAnythingV2(encoder=encoder or "vitb")
            state = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state:
                state = state["model"]
            if "state_dict" in state:
                state = state["state_dict"]
            self._local_model.load_state_dict(state)
            self._local_model.to(device_str)
            self._local_model.eval()
            self._device = device_str
            self._transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self._pipeline = pipeline("depth-estimation", model=model_id, device=device)

    def estimate(self, image: Image.Image) -> DepthResult:
        if self._local_model is not None:
            depth_np = self._estimate_local(image)
        else:
            output = self._pipeline(image)
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

    def _estimate_local(self, image: Image.Image) -> np.ndarray:
        assert self._local_model is not None
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._local_model(tensor)
        if isinstance(output, dict):
            depth = output.get("depth") or output.get("metric_depth") or next(iter(output.values()))
        elif isinstance(output, (list, tuple)):
            depth = output[0]
        else:
            depth = output
        if isinstance(depth, torch.Tensor):
            depth_np = depth.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            depth_np = np.array(depth)
        return depth_np


__all__ = ["DepthAnythingEstimator", "DepthResult"]
