"""SAM 2 based instance segmentation wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sam2.build_sam2 import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "SAM 2 dependencies are missing. Install the official sam2 package."
    ) from exc


@dataclass
class InstanceMask:
    """Associates a binary mask with metadata."""

    mask: np.ndarray
    instance_id: int
    category: Optional[str]
    score: Optional[float] = None


class Sam2Segmenter:
    """Wrapper around SAM 2 predictors for image and video segmentation."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        image_size: int = 1024,
        multimask_output: bool = False,
    ) -> None:
        self.model = build_sam2(
            config_path,
            checkpoint_path,
            device=device,
            image_size=image_size,
        )
        self.image_predictor = SAM2ImagePredictor(self.model)
        self.video_predictor = SAM2VideoPredictor(self.model)
        self.multimask_output = bool(multimask_output)
        self._video_initialized = False

    def segment_image(
        self,
        image: np.ndarray,
        prompts: Sequence[Dict[str, np.ndarray]],
        categories: Optional[Sequence[str]] = None,
        scores: Optional[Sequence[float]] = None,
    ) -> List[InstanceMask]:
        """Segment instances on an RGB image using SAM 2."""

        self.image_predictor.set_image(image)
        masks: List[InstanceMask] = []
        categories = categories or [None] * len(prompts)
        scores = scores or [None] * len(prompts)

        for idx, (prompt, category, score) in enumerate(zip(prompts, categories, scores)):
            prediction = self.image_predictor.predict(
                multimask_output=self.multimask_output,
                **prompt,
            )
            mask_stack = prediction[0]
            mask_scores = prediction[1]
            best_idx = int(np.argmax(mask_scores))
            best_mask = mask_stack[best_idx]
            masks.append(
                InstanceMask(
                    mask=best_mask.astype(bool),
                    instance_id=idx + 1,
                    category=category,
                    score=float(mask_scores[best_idx]) if score is None else score,
                )
            )
        return masks

    def initialize_video(
        self,
        first_frame: np.ndarray,
        prompts: Sequence[Dict[str, np.ndarray]],
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the SAM 2 video predictor with prompts on the first frame."""

        self.video_predictor.reset_state()
        self.video_predictor.set_image(first_frame, frame_idx=0)
        categories = categories or [None] * len(prompts)
        for prompt, category in zip(prompts, categories):
            self.video_predictor.add_new_mask(frame_idx=0, category=category, **prompt)
        self._video_initialized = True

    def segment_video_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> List[InstanceMask]:
        """Predict masks for a subsequent frame using the video predictor."""

        if not self._video_initialized:
            raise RuntimeError("initialize_video must be called before segment_video_frame")

        predictions = self.video_predictor.propagate(frame, frame_idx=frame_idx)
        masks: List[InstanceMask] = []
        for prediction in predictions:
            masks.append(
                InstanceMask(
                    mask=prediction["mask"].astype(bool),
                    instance_id=prediction["instance_id"],
                    category=prediction.get("category"),
                    score=prediction.get("iou_score"),
                )
            )
        return masks


__all__ = ["Sam2Segmenter", "InstanceMask"]
