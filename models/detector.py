"""Detector and prompt generation utilities leveraging DINOv3 + Grounding DINO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The detector module requires transformers with Grounding DINO support."
    ) from exc


BBox = Tuple[float, float, float, float]


@dataclass
class InstanceProposal:
    """A lightweight description of a detected instance."""

    bbox: BBox
    score: float
    category: str
    prompt_type: str = "box"

    def as_prompt(self) -> Dict[str, np.ndarray]:
        """Return a SAM 2 compatible prompt dictionary."""
        if self.prompt_type == "box":
            # SAM 2 expects (x1, y1, x2, y2)
            return {"boxes": np.asarray([self.bbox], dtype=np.float32)}
        if self.prompt_type == "point":
            x1, y1, x2, y2 = self.bbox
            point = np.asarray([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], dtype=np.float32)
            return {"points": point, "labels": np.ones((1,), dtype=np.int32)}
        raise ValueError(f"Unsupported prompt type: {self.prompt_type}")


class Detector:
    """Open-set zero-shot detector using Grounding DINO prompts."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        category_prompts: Optional[Sequence[str]] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.max_detections = max_detections
        self.category_prompts = list(category_prompts or ["object"])

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompts: Optional[Iterable[str]] = None,
    ) -> List[InstanceProposal]:
        """Run the detector on a PIL image and return instance proposals."""

        prompts = list(text_prompts or self.category_prompts)
        prompt_str = ". ".join(prompts)
        inputs = self.processor(images=image, text=prompt_str, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )[0]

        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results["labels"]

        proposals: List[InstanceProposal] = []
        for bbox, score, label in zip(boxes, scores, labels):
            label_str = str(label)
            proposals.append(
                InstanceProposal(
                    bbox=tuple(map(float, bbox)),
                    score=float(score),
                    category=label_str,
                    prompt_type="box",
                )
            )

        proposals.sort(key=lambda p: p.score, reverse=True)
        if self.max_detections is not None:
            proposals = proposals[: self.max_detections]
        return proposals

    def generate_sam_prompts(self, proposals: Sequence[InstanceProposal]) -> List[Dict[str, np.ndarray]]:
        """Convert proposals to SAM 2 compatible prompts."""

        prompts = []
        for proposal in proposals:
            prompts.append(proposal.as_prompt())
        return prompts


__all__ = ["Detector", "InstanceProposal"]
