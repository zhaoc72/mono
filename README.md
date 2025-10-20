# Mono3D Pipeline

This repository provides a lightweight runner that stitches together simplified
implementations inspired by
[DINOv3](https://github.com/facebookresearch/dinov3),
[SAM 2](https://github.com/facebookresearch/sam2), and
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for feature
extraction, segmentation, and monocular depth estimation on Pix3D samples. All
necessary model code is embedded directly in this repository, so no external git
checkouts are required.

## Features

* Automatically discovers Pix3D images (defaults to `/media/pc/D/datasets/pix3d`).
* Generates segmentation overlays using DINOv3 features with SAM 2.
* Produces instance-aware depth renderings with Depth Anything V2.
* Supports single-image and video processing, exporting PNG/MP4 visualizations.
* Reports throughput (FPS) for segmentation-only and segmentation + depth pipelines.

## Usage

## Setup

1. **Prepare a Python environment** and install the runtime dependencies. The exact
   steps depend on your CUDA setup, but one common pattern is:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   # Install PyTorch matching your CUDA version (example shown for CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # Install supporting packages used by the runner
   pip install opencv-python numpy
   ```

   The pipeline loads checkpoints from the paths configured in `mono/pipeline.py`. If a
   checkpoint file is missing the runner will continue with randomly initialised
   weights and emit a warning.

## Running the pipeline

### Pix3D sweep (segmentation + depth visualizations)

The command below iterates through Pix3D images at
`/media/pc/D/datasets/pix3d`, stores segmentation overlays under
`outputs/segmentation/`, depth maps under `outputs/depth/`, and prints the per-stage
FPS to the console:

```bash
python run_pipeline.py \
    --pix3d-root /media/pc/D/datasets/pix3d \
    --limit 10
```

Remove `--limit` to process the full dataset. The default checkpoint paths used by the
pipeline already point to:

* DINOv3 weights at `/media/pc/D/zhaochen/mono3d/checkpoints/pretrained/dinov3_vitl16_lvd1689m.pth`
* SAM 2 weights at `/media/pc/D/zhaochen/mono3d/checkpoints/pretrained/sam2.1_hiera_large.pt`
* Depth Anything V2 weights at `/media/pc/D/zhaochen/mono3d/checkpoints/pretrained/depth_anything_v2_vitl.pth`

### Single image or video

To generate segmentation and depth visualizations for a specific asset (for example,
`sample.jpg`), run:

```bash
python run_pipeline.py --input-path /path/to/sample.jpg
```

Replace `sample.jpg` with the image or video you want to process. The pipeline reports
segmentation FPS (DINOv3 + SAM 2) and depth FPS (DINOv3 + SAM 2 + Depth Anything V2)
after each run, and visualizations are exported to the same output directories noted
above.