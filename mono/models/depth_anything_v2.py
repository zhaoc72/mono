"""Minimal Depth Anything V2-style depth estimator."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mono.utils.logging import get_logger

LOGGER = get_logger(__name__)


class DummyDepthAnythingV2(nn.Module):
    """Produces a pseudo-depth map using grayscale intensity gradients."""

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", kernel)
        self.register_buffer("sobel_y", kernel.transpose(-1, -2))

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: Path | None = None, device: str = "cpu"
    ) -> "DummyDepthAnythingV2":
        model = cls()
        if checkpoint_path is not None and not Path(checkpoint_path).exists():
            LOGGER.warning("DepthAnything V2 checkpoint not found at %s", checkpoint_path)
        model.eval().to(device)
        return model

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(self.sobel_x.device).float()
        r, g, b = image_tensor[:, 0:1], image_tensor[:, 1:2], image_tensor[:, 2:3]
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
        grad_x = F.conv2d(grayscale, self.sobel_x, padding=1)
        grad_y = F.conv2d(grayscale, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        depth = F.avg_pool2d(magnitude, kernel_size=3, stride=1, padding=1)
        min_val = depth.amin(dim=(1, 2, 3), keepdim=True)
        max_val = depth.amax(dim=(1, 2, 3), keepdim=True)
        normalized = (depth - min_val) / (max_val - min_val + 1e-6)
        return normalized

