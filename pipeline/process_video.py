"""Video processing pipeline leveraging SAM 2 streaming."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    timings: Dict[str, Dict[str, float]]


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

        initialized = False
        for idx, frame in enumerate(frames):
            if not initialized:
                first_image = Image.fromarray(frame)
                proposals = self.detector.detect(first_image)
                prompts = self.detector.generate_sam_prompts(proposals)
                categories = [proposal.category for proposal in proposals]
                self.segmenter.initialize_video(frame, prompts, categories=categories)
                initialized = True

            with ThreadPoolExecutor(max_workers=2) as executor:
                seg_future = executor.submit(self._segment_video_frame_task, frame, idx)
                depth_future = executor.submit(self._depth_task, frame)
                mask_predictions, seg_time = seg_future.result()
                depth_result, depth_time = depth_future.result()

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

            metadata = {
                "intrinsics": intrinsics,
                "num_instances": len(mask_predictions),
                "timings": {
                    "segmentation": {
                        "time": float(seg_time),
                        "fps": float(1.0 / seg_time) if seg_time > 0 else 0.0,
                        "num_masks": len(mask_predictions),
                    },
                    "depth": {
                        "time": float(depth_time),
                        "fps": float(1.0 / depth_time) if depth_time > 0 else 0.0,
                    },
                },
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
                    timings=metadata["timings"],
                )
            )
        return frame_results

    def _segment_video_frame_task(self, frame: np.ndarray, frame_idx: int):
        import time

        start = time.perf_counter()
        masks = self.segmenter.segment_video_frame(frame, frame_idx=frame_idx)
        elapsed = time.perf_counter() - start
        return masks, elapsed

    def _depth_task(self, frame: np.ndarray):
        import time

        start = time.perf_counter()
        depth_result = self.depth_estimator.estimate(Image.fromarray(frame))
        elapsed = time.perf_counter() - start
        return depth_result, elapsed
