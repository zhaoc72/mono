"""Command-line interface for running the Mono3D pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from mono.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv3 + SAM2 + DepthAnything V2")
    parser.add_argument(
        "--pix3d-root",
        type=Path,
        default=None,
        help="Path to the Pix3D dataset root (defaults to configured path)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional image size to resize inputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of Pix3D images to process",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional path to a single image or video to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device identifier (e.g., cuda or cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(
        pix3d_root=args.pix3d_root,
        image_size=tuple(args.image_size) if args.image_size else None,
        device=args.device,
    )
    pipeline.run(input_path=args.input_path, limit=args.limit)


if __name__ == "__main__":
    main()