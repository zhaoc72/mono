"""Single frame processing pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image

from models.depth_estimator import DepthAnythingEstimator, DepthResult
from models.detector import Detector
from models.segmenter import InstanceMask, Sam2Segmenter
from utils.geometry import depth_to_point_cloud


@dataclass
class FrameResult:
    image: np.ndarray
    depth: np.ndarray
    instance_id_map: np.ndarray
    instances: Dict[int, Dict]
    masks: Dict[int, InstanceMask]
    metadata: Dict
    depth_result: DepthResult
    timings: Dict[str, Dict[str, float]]


class FrameProcessor:
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

    def _build_instance_map(self, image_shape: tuple, masks: Dict[int, InstanceMask]) -> np.ndarray:
        instance_map = np.zeros(image_shape[:2], dtype=np.uint16)
        for instance_id, instance_mask in masks.items():
            instance_map[instance_mask.mask] = instance_id
        return instance_map

    def process(
        self,
        image: Image.Image,
        intrinsics: Dict[str, float],
    ) -> FrameResult:
        proposals = self.detector.detect(image)
        prompts = self.detector.generate_sam_prompts(proposals)
        categories = [proposal.category for proposal in proposals]
        scores = [proposal.score for proposal in proposals]

        np_image = np.array(image, dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=2) as executor:
            seg_future = executor.submit(
                self._segment_image_task,
                np_image,
                prompts,
                categories,
                scores,
            )
            depth_future = executor.submit(self._depth_task, image)
            mask_list, seg_time = seg_future.result()
            depth_result, depth_time = depth_future.result()

        mask_dict = {mask.instance_id: mask for mask in mask_list}

        instance_map = self._build_instance_map(np_image.shape, mask_dict)

        instances_metadata = {}
        for mask, proposal in zip(mask_list, proposals):
            ys, xs = np.where(mask.mask)
            if ys.size:
                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            else:
                bbox = list(proposal.bbox)
            instances_metadata[mask.instance_id] = {
                "category": mask.category,
                "score": mask.score,
                "bbox": bbox,
            }

        timings = {
            "segmentation": {
                "time": float(seg_time),
                "fps": float(1.0 / seg_time) if seg_time > 0 else 0.0,
                "num_masks": len(mask_list),
            },
            "depth": {
                "time": float(depth_time),
                "fps": float(1.0 / depth_time) if depth_time > 0 else 0.0,
            },
        }

        metadata = {
            "intrinsics": intrinsics,
            "num_instances": len(mask_list),
            "timings": timings,
        }

        geometry_cfg = self.config.get("pipeline", {}).get("geometry", {})
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

        return FrameResult(
            image=np_image,
            depth=depth_result.depth,
            instance_id_map=instance_map,
            instances=instances_metadata,
            masks=mask_dict,
            metadata=metadata,
            depth_result=depth_result,
            timings=timings,
        )

    def _segment_image_task(
        self,
        image: np.ndarray,
        prompts,
        categories,
        scores,
    ):
        import time

        start = time.perf_counter()
        masks = self.segmenter.segment_image(image, prompts, categories=categories, scores=scores)
        elapsed = time.perf_counter() - start
        return masks, elapsed

    def _depth_task(self, image: Image.Image):
        import time

        start = time.perf_counter()
        depth_result = self.depth_estimator.estimate(image)
        elapsed = time.perf_counter() - start
        return depth_result, elapsed
