"""Detector and prompt generation utilities leveraging DINOv3 + Grounding DINO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForZeroShotObjectDetection = None
    AutoProcessor = None

from .dino_features import DINOv3FeatureExtractor


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
    """Open-set zero-shot detector using Grounding DINO and DINOv3 cues."""

    def __init__(
        self,
        model_id: Optional[str],
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        category_prompts: Optional[Sequence[str]] = None,
        max_detections: Optional[int] = None,
        feature_extractor: Optional[DINOv3FeatureExtractor] = None,
        unsupervised_topk: int = 10,
        unsupervised_threshold: float = 0.6,
        unsupervised_box_scale: float = 2.5,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = feature_extractor
        self.unsupervised_topk = int(unsupervised_topk)
        self.unsupervised_threshold = float(unsupervised_threshold)
        self.unsupervised_box_scale = float(unsupervised_box_scale)

        self.model_id = model_id
        if model_id:
            if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
                raise ImportError(
                    "transformers with Grounding DINO support is required for hf_model_id."
                )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
        else:
            self.processor = None
            self.model = None
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
        proposals: List[InstanceProposal] = []

        if self.model is not None and self.processor is not None:
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

        unsupervised = self._generate_unsupervised_proposals(np.array(image))
        proposals = self._merge_proposals(proposals, unsupervised)
        return proposals

    def generate_sam_prompts(self, proposals: Sequence[InstanceProposal]) -> List[Dict[str, np.ndarray]]:
        """Convert proposals to SAM 2 compatible prompts."""

        prompts = []
        for proposal in proposals:
            prompts.append(proposal.as_prompt())
        return prompts

    def _generate_unsupervised_proposals(self, image: np.ndarray) -> List[InstanceProposal]:
        if self.feature_extractor is None:
            return []

        dense = self.feature_extractor.extract_dense(image)
        feat = dense.features
        saliency = np.linalg.norm(feat, axis=-1)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
        flat = saliency.reshape(-1)
        order = np.argsort(flat)[::-1]

        h_tokens, w_tokens = feat.shape[:2]
        height, width = image.shape[:2]
        stride = dense.stride

        boxes: List[InstanceProposal] = []
        taken: List[Tuple[float, float, float, float]] = []
        for idx in order[: self.unsupervised_topk * 5]:
            score = float(flat[idx])
            if score < self.unsupervised_threshold:
                continue
            token_y, token_x = divmod(int(idx), w_tokens)
            cx = (token_x + 0.5) * stride
            cy = (token_y + 0.5) * stride
            half_w = stride * self.unsupervised_box_scale * 0.5
            half_h = stride * self.unsupervised_box_scale * 0.5
            x1 = float(np.clip(cx - half_w, 0, width - 1))
            y1 = float(np.clip(cy - half_h, 0, height - 1))
            x2 = float(np.clip(cx + half_w, 0, width - 1))
            y2 = float(np.clip(cy + half_h, 0, height - 1))
            bbox = (x1, y1, x2, y2)
            if any(self._bbox_iou(bbox, existing) > 0.6 for existing in taken):
                continue
            taken.append(bbox)
            boxes.append(
                InstanceProposal(
                    bbox=bbox,
                    score=score,
                    category="object",
                    prompt_type="box",
                )
            )
            if len(boxes) >= self.unsupervised_topk:
                break
        return boxes

    @staticmethod
    def _bbox_iou(box_a: BBox, box_b: BBox) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area + 1e-6
        return inter_area / union

    def _merge_proposals(
        self,
        grounding: List[InstanceProposal],
        unsupervised: List[InstanceProposal],
    ) -> List[InstanceProposal]:
        proposals = list(grounding)
        existing_boxes = [proposal.bbox for proposal in proposals]
        for proposal in unsupervised:
            if any(self._bbox_iou(proposal.bbox, box) > 0.7 for box in existing_boxes):
                continue
            proposals.append(proposal)
            existing_boxes.append(proposal.bbox)
        proposals.sort(key=lambda p: p.score, reverse=True)
        if self.max_detections is not None:
            proposals = proposals[: self.max_detections]
        return proposals


__all__ = ["Detector", "InstanceProposal"]
