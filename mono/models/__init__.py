"""Lightweight stand-ins for external research models."""
from .dinov3 import DummyDINOv3
from .sam2 import DummySAM2
from .depth_anything_v2 import DummyDepthAnythingV2

__all__ = ["DummyDINOv3", "DummySAM2", "DummyDepthAnythingV2"]
