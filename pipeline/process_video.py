"""Video processing pipeline leveraging SAM 2 streaming."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image

from models.depth_estimator import DepthAnythingEstimator, DepthResult
from models.detector import Detector
from models.segmenter import InstanceMask, Sam2Segmenter
from utils.geometry import depth_to_point_cloud


@dataclass
class VideoFrameResult:
    frame_idx: int
    image: np.ndarray
    depth: np.ndarray
    instance_id_map: np.ndarray
    instances: Dict[int, Dict]
    masks: Dict[int, InstanceMask]
    metadata: Dict
    depth_result: DepthResult


class VideoProcessor:
    def __init__(
        self,
        detector: Detector,
        segmenter: Sam2Segmenter,
        depth_estimator: DepthAnythingEstimator,
        config: Dict,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.config = config

    def process(
        self,
        frames: Iterable[np.ndarray],
        intrinsics: Dict[str, float],
    ) -> List[VideoFrameResult]:
        frame_results: List[VideoFrameResult] = []
        geometry_cfg = self.config.get("pipeline", {}).get("geometry", {})

        first_frame = None
        for idx, frame in enumerate(frames):
            if first_frame is None:
                first_frame = frame
                first_image = Image.fromarray(frame)
                proposals = self.detector.detect(first_image)
                prompts = self.detector.generate_sam_prompts(proposals)
                categories = [proposal.category for proposal in proposals]
                self.segmenter.initialize_video(frame, prompts, categories=categories)
                mask_predictions = self.segmenter.segment_video_frame(frame, frame_idx=0)
            else:
                mask_predictions = self.segmenter.segment_video_frame(frame, frame_idx=idx)

            instance_map = np.zeros(frame.shape[:2], dtype=np.uint16)
            instances_metadata: Dict[int, Dict] = {}
            mask_dict: Dict[int, InstanceMask] = {}
            for mask in mask_predictions:
                instance_map[mask.mask] = mask.instance_id
                mask.mask = mask.mask.astype(bool)
                mask_dict[mask.instance_id] = mask
                ys, xs = np.where(mask.mask)
                if ys.size:
                    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                instances_metadata[mask.instance_id] = {
                    "category": mask.category,
                    "score": mask.score,
                    "bbox": bbox,
                }

            depth_result = self.depth_estimator.estimate(Image.fromarray(frame))
            metadata = {
                "intrinsics": intrinsics,
                "num_instances": len(mask_predictions),
            }

            if geometry_cfg.get("point_cloud", False):
                points, pixels = depth_to_point_cloud(
                    depth_result.depth,
                    intrinsics,
                    mask=None,
                    max_points=geometry_cfg.get("max_points"),
                )
                metadata["point_cloud_preview"] = {
                    "points": points.tolist(),
                    "pixels": pixels.tolist(),
                }

            frame_results.append(
                VideoFrameResult(
                    frame_idx=idx,
                    image=frame,
                    depth=depth_result.depth,
                    instance_id_map=instance_map,
                    instances=instances_metadata,
                    masks=mask_dict,
                    metadata=metadata,
                    depth_result=depth_result,
                )
            )
        return frame_results
