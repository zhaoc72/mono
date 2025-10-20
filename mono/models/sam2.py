"""Minimal SAM2-style segmenter used within the pipeline."""
from __future__ import annotations

from pathlib import Path

import torch

from mono.utils.logging import get_logger

LOGGER = get_logger(__name__)


class DummySAM2:
    """Simple mask predictor that thresholds encoder activations."""

    def __init__(self, threshold: float = 0.5, device: str = "cpu") -> None:
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: Path | None = None, device: str = "cpu"
    ) -> "DummySAM2":
        if checkpoint_path is not None and not Path(checkpoint_path).exists():
            LOGGER.warning("SAM2 checkpoint not found at %s", checkpoint_path)
        return cls(device=device)

    def predict_masks_from_embeddings(
        self, image_tensor: torch.Tensor, image_embeddings: torch.Tensor | list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if isinstance(image_embeddings, (list, tuple)):
            embeddings = image_embeddings[0]
        else:
            embeddings = image_embeddings
        embeddings = embeddings.to(self.device)
        if embeddings.dim() == 4:
            embeddings = embeddings.mean(dim=1)
        elif embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Unexpected embedding shape for SAM2 dummy: {tuple(embeddings.shape)}"
            )
        embeddings = embeddings.squeeze(0)
        embeddings = embeddings - embeddings.min()
        max_val = embeddings.max()
        if float(max_val) > 1e-6:
            embeddings = embeddings / max_val
        mask = (embeddings > self.threshold).float().unsqueeze(0)
        return {"masks": mask.cpu()}

