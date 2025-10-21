"""I/O helpers for reading inputs and writing structured outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import cv2
import numpy as np
from PIL import Image


@dataclass
class FrameOutput:
    image_path: Path
    depth_path: Path
    mask_path: Path
    metadata_path: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image, dtype=np.uint8)


def write_image(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(array).save(path)


def write_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(mask.astype(np.uint16)).save(path)


def write_depth(path: Path, depth: np.ndarray) -> None:
    ensure_dir(path.parent)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    np.save(path, depth)


def write_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def iter_video_frames(video_path: Path) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def frame_output_paths(output_root: Path, frame_idx: int) -> FrameOutput:
    frame_dir = output_root / f"frame_{frame_idx:05d}"
    return FrameOutput(
        image_path=frame_dir / "image.png",
        depth_path=frame_dir / "depth.npy",
        mask_path=frame_dir / "instance_ids.png",
        metadata_path=frame_dir / "data.json",
    )
