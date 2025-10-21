"""Command line entry point for the zero-shot 2D+depth pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from models.depth_estimator import DepthAnythingEstimator
from models.detector import Detector
from models.dino_features import DINOv3FeatureExtractor
from models.segmenter import Sam2Segmenter
from pipeline.process_frame import FrameProcessor
from pipeline.process_video import VideoProcessor
from utils import io as io_utils
from utils.visualization import draw_depth_heatmap, draw_instance_masks


def build_models(config: Dict[str, Any]):
    device = config.get("device")

    dino_cfg = config["models"].get("dinov3")
    feature_extractor = None
    if dino_cfg and dino_cfg.get("checkpoint"):
        feature_extractor = DINOv3FeatureExtractor(
            checkpoint_path=dino_cfg["checkpoint"],
            model_name=dino_cfg.get("model_name", "vit_large_patch16_224.dino"),
            image_mean=tuple(dino_cfg.get("image_mean", (0.485, 0.456, 0.406))),
            image_std=tuple(dino_cfg.get("image_std", (0.229, 0.224, 0.225))),
            device=device,
        )

    detector_cfg = config["models"]["detector"]
    detector = Detector(
        model_id=detector_cfg.get("hf_model_id"),
        device=device,
        box_threshold=detector_cfg.get("box_threshold", 0.25),
        text_threshold=detector_cfg.get("text_threshold", 0.25),
        category_prompts=detector_cfg.get("category_prompts"),
        max_detections=detector_cfg.get("max_detections"),
        feature_extractor=feature_extractor,
        unsupervised_topk=detector_cfg.get("unsupervised_topk", 10),
        unsupervised_threshold=detector_cfg.get("unsupervised_threshold", 0.6),
        unsupervised_box_scale=detector_cfg.get("unsupervised_box_scale", 2.5),
    )

    segmenter_cfg = config["models"]["segmenter"]
    segmenter = Sam2Segmenter(
        config_path=segmenter_cfg["model_config"],
        checkpoint_path=segmenter_cfg["checkpoint"],
        device=device,
        image_size=segmenter_cfg.get("image_size", 1024),
        multimask_output=segmenter_cfg.get("multimask_output", False),
    )

    depth_cfg = config["models"]["depth"]
    if device and str(device).startswith("cuda"):
        depth_device = 0
    else:
        depth_device = None
    depth_estimator = DepthAnythingEstimator(
        model_id=depth_cfg["hf_model_id"],
        device=depth_device,
        normalize_depth=depth_cfg.get("normalize_depth", True),
        checkpoint=depth_cfg.get("checkpoint"),
        encoder=depth_cfg.get("encoder"),
    )

    return detector, segmenter, depth_estimator


def intrinsics_from_config(config: Dict[str, Any], dataset: str) -> Dict[str, float]:
    intrinsics_cfg = config.get("datasets", {}).get("intrinsics", {})
    if dataset in intrinsics_cfg:
        return intrinsics_cfg[dataset]
    return intrinsics_cfg.get("default", {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0})



def process_image(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    detector, segmenter, depth_estimator = build_models(config)
    processor = FrameProcessor(detector, segmenter, depth_estimator, config)

    image_path = Path(args.input)
    image = io_utils.load_image(image_path)
    intrinsics = intrinsics_from_config(config, args.dataset)

    result = processor.process(image, intrinsics)

    timings = result.timings
    print(
        f"Segmentation FPS: {timings['segmentation']['fps']:.2f} | "
        f"Depth FPS: {timings['depth']['fps']:.2f}"
    )

    output_root = Path(config["pipeline"]["output_root"]).resolve()
    frame_paths = io_utils.frame_output_paths(output_root, 0)
    io_utils.ensure_dir(frame_paths.image_path.parent)

    io_utils.write_image(frame_paths.image_path, result.image)
    io_utils.write_depth(frame_paths.depth_path, result.depth)
    io_utils.write_mask(frame_paths.mask_path, result.instance_id_map)

    visualization_cfg = config["pipeline"].get("visualization", {})
    overlay = draw_instance_masks(
        result.image,
        list(result.masks.values()),
        draw_labels=visualization_cfg.get("draw_labels", True),
        draw_scores=visualization_cfg.get("draw_scores", True),
    )
    depth_vis = draw_depth_heatmap(
        result.depth_result,
        cmap=visualization_cfg.get("depth_colormap", "plasma"),
    )

    io_utils.write_image(frame_paths.image_path.with_name("overlay_seg.png"), overlay)
    io_utils.write_image(frame_paths.image_path.with_name("overlay_depth.png"), depth_vis)

    instance_list = []
    for instance_id, meta in result.instances.items():
        instance_entry = {"id": int(instance_id), **meta}
        instance_list.append(instance_entry)

    metadata = dict(result.metadata)
    metadata["instances"] = instance_list
    metadata["instance_id_map"] = str(frame_paths.mask_path)
    metadata["depth_map"] = str(frame_paths.depth_path)
    metadata["image"] = str(frame_paths.image_path)
    io_utils.write_json(frame_paths.metadata_path, metadata)



def process_video(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    detector, segmenter, depth_estimator = build_models(config)
    processor = VideoProcessor(detector, segmenter, depth_estimator, config)

    video_path = Path(args.input)
    frames = list(io_utils.iter_video_frames(video_path))
    intrinsics = intrinsics_from_config(config, args.dataset)

    results = processor.process(frames, intrinsics)

    output_root = Path(config["pipeline"]["output_root"]).resolve()
    for frame_result in results:
        paths = io_utils.frame_output_paths(output_root, frame_result.frame_idx)
        io_utils.ensure_dir(paths.image_path.parent)
        io_utils.write_image(paths.image_path, frame_result.image)
        io_utils.write_depth(paths.depth_path, frame_result.depth)
        io_utils.write_mask(paths.mask_path, frame_result.instance_id_map)

        visualization_cfg = config["pipeline"].get("visualization", {})
        overlay = draw_instance_masks(
            frame_result.image,
            list(frame_result.masks.values()),
            draw_labels=visualization_cfg.get("draw_labels", True),
            draw_scores=visualization_cfg.get("draw_scores", True),
        )
        depth_vis = draw_depth_heatmap(
            frame_result.depth_result,
            cmap=visualization_cfg.get("depth_colormap", "plasma"),
        )

        io_utils.write_image(paths.image_path.with_name("overlay_seg.png"), overlay)
        io_utils.write_image(paths.image_path.with_name("overlay_depth.png"), depth_vis)

        instance_list = []
        for instance_id, meta in frame_result.instances.items():
            instance_entry = {"id": int(instance_id), **meta}
            instance_list.append(instance_entry)

        print(
            f"[frame {frame_result.frame_idx:05d}] Segmentation FPS: "
            f"{frame_result.timings['segmentation']['fps']:.2f} | Depth FPS: "
            f"{frame_result.timings['depth']['fps']:.2f}"
        )

        metadata = dict(frame_result.metadata)
        metadata["instances"] = instance_list
        metadata["instance_id_map"] = str(paths.mask_path)
        metadata["depth_map"] = str(paths.depth_path)
        metadata["image"] = str(paths.image_path)
        io_utils.write_json(paths.metadata_path, metadata)


def process_folder(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    detector, segmenter, depth_estimator = build_models(config)
    processor = FrameProcessor(detector, segmenter, depth_estimator, config)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    image_paths = sorted(
        p
        for p in input_path.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found under {input_path}")

    intrinsics = intrinsics_from_config(config, args.dataset)
    output_root = Path(config["pipeline"]["output_root"]).resolve()

    for idx, image_path in enumerate(image_paths):
        image = io_utils.load_image(image_path)
        result = processor.process(image, intrinsics)
        timings = result.timings
        print(
            f"[{image_path.name}] Segmentation FPS: {timings['segmentation']['fps']:.2f} | "
            f"Depth FPS: {timings['depth']['fps']:.2f}"
        )

        frame_paths = io_utils.frame_output_paths(output_root, idx)
        io_utils.ensure_dir(frame_paths.image_path.parent)
        io_utils.write_image(frame_paths.image_path, result.image)
        io_utils.write_depth(frame_paths.depth_path, result.depth)
        io_utils.write_mask(frame_paths.mask_path, result.instance_id_map)

        visualization_cfg = config["pipeline"].get("visualization", {})
        overlay = draw_instance_masks(
            result.image,
            list(result.masks.values()),
            draw_labels=visualization_cfg.get("draw_labels", True),
            draw_scores=visualization_cfg.get("draw_scores", True),
        )
        depth_vis = draw_depth_heatmap(
            result.depth_result,
            cmap=visualization_cfg.get("depth_colormap", "plasma"),
        )

        io_utils.write_image(frame_paths.image_path.with_name("overlay_seg.png"), overlay)
        io_utils.write_image(frame_paths.image_path.with_name("overlay_depth.png"), depth_vis)

        instance_list = []
        for instance_id, meta in result.instances.items():
            instance_entry = {"id": int(instance_id), **meta}
            instance_list.append(instance_entry)

        metadata = dict(result.metadata)
        metadata["instances"] = instance_list
        metadata["instance_id_map"] = str(frame_paths.mask_path)
        metadata["depth_map"] = str(frame_paths.depth_path)
        metadata["image"] = str(frame_paths.image_path)
        metadata["source"] = str(image_path)
        io_utils.write_json(frame_paths.metadata_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--input", type=str, required=True, help="Path to an input image, video, or folder")
    parser.add_argument("--mode", type=str, choices=["image", "video", "folder"], default="image")
    parser.add_argument("--dataset", type=str, default="default", help="Dataset name for intrinsics lookup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("pipeline", {})["mode"] = args.mode

    if args.mode == "image":
        process_image(args, config)
    elif args.mode == "video":
        process_video(args, config)
    else:
        process_folder(args, config)


if __name__ == "__main__":
    main()
