"""DINOv3 feature extraction utilities for prompt generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import timm
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "DINOv3 feature extraction requires torch, timm and torchvision."
    ) from exc


@dataclass
class DenseFeatureMap:
    """Container for dense DINOv3 features."""

    features: np.ndarray
    stride: int

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return self.features.shape[:2]


class DINOv3FeatureExtractor:
    """Load a ViT backbone trained with DINOv3 and expose dense features."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vit_large_patch16_224.dino",
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:  # pragma: no cover - informative logging path
            print(f"[DINOv3] Missing keys: {missing}")
        if unexpected:  # pragma: no cover - informative logging path
            print(f"[DINOv3] Unexpected keys: {unexpected}")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )

        patch_size = getattr(self.model, "patch_embed").patch_size
        if isinstance(patch_size, tuple):
            self.patch_stride = int(patch_size[0])
        else:
            self.patch_stride = int(patch_size)

    @torch.no_grad()
    def extract_dense(self, image: np.ndarray) -> DenseFeatureMap:
        """Return dense features for an RGB image."""

        if image.ndim != 3:
            raise ValueError("Expected an RGB image array with shape (H, W, 3)")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(enabled=tensor.device.type == "cuda"):
            features = self.model.forward_features(tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            features = features.get("x_norm_patchtokens") or features.get("x") or features.get("last_hidden_state")
        if features.ndim == 3:
            features = features[:, 1:, :]
        b, n, c = features.shape
        h = w = int(n ** 0.5)
        features = features.reshape(b, h, w, c)
        features_np = features.squeeze(0).detach().cpu().numpy()
        return DenseFeatureMap(features=features_np, stride=self.patch_stride)


__all__ = ["DINOv3FeatureExtractor", "DenseFeatureMap"]
