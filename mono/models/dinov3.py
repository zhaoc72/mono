"""Minimal DINOv3-style feature extractor used within the pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from mono.utils.logging import get_logger

LOGGER = get_logger(__name__)


class DummyDINOv3(nn.Module):
    """A lightweight convolutional backbone that mimics DINOv3 interfaces."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: Path | None = None, device: str = "cpu"
    ) -> "DummyDINOv3":
        model = cls()
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            if path.exists():
                try:
                    state_dict = torch.load(path, map_location="cpu")
                    if isinstance(state_dict, dict):
                        missing, unexpected = model.load_state_dict(
                            state_dict, strict=False
                        )
                        if missing or unexpected:
                            LOGGER.warning(
                                "Loaded checkpoint with missing=%s unexpected=%s",
                                missing,
                                unexpected,
                            )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to load DINOv3 checkpoint: %s", exc)
            else:
                LOGGER.warning("DINOv3 checkpoint not found at %s", path)
        model.eval().to(device)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.encoder(x)

    def get_intermediate_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Match the API expected by the original DINOv3 integration."""
        return [self.forward(x)]

