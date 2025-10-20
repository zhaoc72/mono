"""Processing pipeline for DINOv3 + SAM2 + DepthAnything V2."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch

from mono.utils.logging import get_logger
from mono.utils.profiling import AverageMeter
from mono.utils.vis import (
    colorize_depth,
    overlay_segmentation,
    save_depth_map,
    save_segmentation_mask,
)
from mono.models import DummyDINOv3, DummyDepthAnythingV2, DummySAM2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

LOGGER = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    pix3d_root: Path = Path("/media/pc/D/datasets/pix3d")
    dinov3_checkpoint: Path = Path(
        "/media/pc/D/zhaochen/mono3d/checkpoints/pretrained/dinov3_vitl16_lvd1689m.pth"
    )
    sam2_checkpoint: Path = Path(
        "/media/pc/D/zhaochen/mono3d/checkpoints/pretrained/sam2.1_hiera_large.pt"
    )
    depthanything_checkpoint: Path = Path(
        "/media/pc/D/zhaochen/mono3d/checkpoints/pretrained/depth_anything_v2_vitl.pth"
    )
    device: str = "cuda"
    segmentation_dir: Path = Path("outputs/segmentation")
    depth_dir: Path = Path("outputs/depth")
    image_size: Tuple[int, int] | None = None


class Mono3DPipeline:
    """End-to-end pipeline connecting DINOv3, SAM2 and DepthAnything V2."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        self.feature_extractor = self._build_feature_extractor()
        self.segmenter = self._build_segmenter()
        self.depth_model = self._build_depth_model()

        self.config.segmentation_dir.mkdir(parents=True, exist_ok=True)
        self.config.depth_dir.mkdir(parents=True, exist_ok=True)

    # region build models
    def _build_feature_extractor(self):
        model = DummyDINOv3.from_pretrained(
            checkpoint_path=self.config.dinov3_checkpoint,
            device=self.config.device,
        )
        LOGGER.info("Initialized lightweight DINOv3 feature extractor")
        return model

    def _build_segmenter(self):
        segmenter = DummySAM2.from_pretrained(
            checkpoint_path=self.config.sam2_checkpoint,
            device=self.config.device,
        )
        LOGGER.info("Initialized lightweight SAM2 segmenter")
        return segmenter

    def _build_depth_model(self):
        model = DummyDepthAnythingV2.from_pretrained(
            checkpoint_path=self.config.depthanything_checkpoint,
            device=self.config.device,
        )
        LOGGER.info("Initialized lightweight Depth Anything V2 model")
        return model

    # endregion

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        processed = image
        if self.config.image_size is not None:
            processed = cv2.resize(processed, self.config.image_size)
        rgb = processed[:, :, ::-1]  # BGR to RGB
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.config.device), processed

    def _extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.feature_extractor.get_intermediate_layers(image_tensor)[0]
        return feats

    def _segment(self, image_tensor: torch.Tensor, features: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            sam_output = self.segmenter.predict_masks_from_embeddings(
                image_tensor=image_tensor,
                image_embeddings=features,
            )
        mask = sam_output["masks"].cpu().numpy()
        combined_mask = mask.max(axis=0)
        return (combined_mask > 0.5).astype(np.uint8)

    def _estimate_depth(self, image_tensor: torch.Tensor, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            depth = self.depth_model(image_tensor)
        depth_np = depth.squeeze().cpu().numpy()
        return depth_np * mask

    def process_image(self, image_path: Path) -> Tuple[float, float]:
        LOGGER.info("Processing image: %s", image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image_tensor, processed_image = self._preprocess_image(image)

        start = time.perf_counter()
        features = self._extract_features(image_tensor)
        mask = self._segment(image_tensor, features)
        segmentation_time = time.perf_counter() - start

        depth_start = time.perf_counter()
        depth = self._estimate_depth(image_tensor, mask)
        depth_time = time.perf_counter() - depth_start

        seg_output_path = self.config.segmentation_dir / f"{image_path.stem}_mask.png"
        depth_output_path = self.config.depth_dir / f"{image_path.stem}_depth.png"

        save_segmentation_mask(processed_image, mask, seg_output_path)
        save_depth_map(depth, depth_output_path)

        return segmentation_time, depth_time

    def process_video(self, video_path: Path) -> Tuple[float, float]:
        LOGGER.info("Processing video: %s", video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        seg_output_path = self.config.segmentation_dir / f"{video_path.stem}_mask.mp4"
        depth_output_path = self.config.depth_dir / f"{video_path.stem}_depth.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        seg_writer = None
        depth_writer = None

        seg_time_accum = 0.0
        depth_time_accum = 0.0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_tensor, processed_frame = self._preprocess_image(frame)
            height, width = processed_frame.shape[:2]

            if seg_writer is None:
                seg_writer = cv2.VideoWriter(
                    str(seg_output_path), fourcc, fps, (width, height)
                )
            if depth_writer is None:
                depth_writer = cv2.VideoWriter(
                    str(depth_output_path), fourcc, fps, (width, height)
                )

            start = time.perf_counter()
            features = self._extract_features(image_tensor)
            mask = self._segment(image_tensor, features)
            seg_time_accum += time.perf_counter() - start

            depth_start = time.perf_counter()
            depth = self._estimate_depth(image_tensor, mask)
            depth_time_accum += time.perf_counter() - depth_start

            overlay = overlay_segmentation(processed_frame, mask)
            depth_vis = colorize_depth(depth)

            seg_writer.write(overlay)
            depth_writer.write(depth_vis)

            frame_count += 1

        cap.release()
        if seg_writer is not None:
            seg_writer.release()
        if depth_writer is not None:
            depth_writer.release()

        if frame_count == 0:
            raise ValueError(f"Video contains no frames: {video_path}")

        return seg_time_accum / frame_count, depth_time_accum / frame_count

    def process_path(self, path: Path) -> Tuple[float, float]:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return self.process_image(path)
        if suffix in VIDEO_EXTENSIONS:
            return self.process_video(path)
        raise ValueError(f"Unsupported file type for path: {path}")

    def iter_dataset(self) -> Iterable[Path]:
        images_dir = self.config.pix3d_root / "img"
        if not images_dir.exists():
            raise FileNotFoundError(f"Pix3D images directory not found: {images_dir}")
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path

    def run(self, input_path: Optional[Path] = None, limit: Optional[int] = None) -> None:
        if input_path is not None:
            seg_time, depth_time = self.process_path(input_path)
            seg_fps = 1.0 / max(seg_time, 1e-6)
            depth_fps = 1.0 / max(depth_time, 1e-6)
            LOGGER.info(
                "Completed %s | Seg FPS: %.2f | Depth FPS: %.2f",
                input_path.name,
                seg_fps,
                depth_fps,
            )
            return

        seg_meter = AverageMeter("Segmentation FPS")
        depth_meter = AverageMeter("Depth FPS")

        for idx, image_path in enumerate(self.iter_dataset()):
            if limit is not None and idx >= limit:
                break
            seg_time, depth_time = self.process_image(image_path)
            seg_meter.update(1.0 / max(seg_time, 1e-6))
            depth_meter.update(1.0 / max(depth_time, 1e-6))
            LOGGER.info(
                "Processed %s | Seg FPS: %.2f | Depth FPS: %.2f",
                image_path.name,
                seg_meter.avg,
                depth_meter.avg,
            )

        LOGGER.info(
            "Finished Pix3D evaluation | Mean Seg FPS: %.2f | Mean Depth FPS: %.2f",
            seg_meter.avg,
            depth_meter.avg,
        )


def build_pipeline(
    pix3d_root: Optional[Path] = None,
    image_size: Optional[Tuple[int, int]] = None,
    device: str = "cuda",
) -> Mono3DPipeline:
    config = PipelineConfig(device=device)
    if pix3d_root is not None:
        config.pix3d_root = pix3d_root
    if image_size is not None:
        config.image_size = image_size
    return Mono3DPipeline(config)