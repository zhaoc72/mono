# Zero-Shot Instance Segmentation and Depth Estimation Pipeline

This repository provides a modular PyTorch pipeline that combines **DINOv3**, **Segment Anything Model 2 (SAM 2)**, and **Depth Anything v2** to perform zero-shot instance segmentation and monocular depth estimation on datasets such as ScanNet, KITTI, vKITTI2, and CO3D-v2. The generated outputs are formatted for downstream **3D Gaussian Splatting (3DGS)** reconstruction workflows and include rich visual diagnostics.

## Key Features

- **Feature Extraction & Prompt Generation** using DINOv3 features coupled with Grounding DINO 1.5 for open-set detection.
- **Instance Segmentation** powered by SAM 2 for both single image and video workflows, including persistent instance ID tracking.
- **Monocular Depth Estimation** with Depth Anything v2 to produce high fidelity relative depth maps.
- **Structured Outputs** bundling RGB frames, camera intrinsics/extrinsics, instance masks, semantic labels, and depth maps.
- **Extensive Visualizations** including segmentation overlays, depth heatmaps, and optional point cloud previews.

## Repository Layout

```
project_root/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── models/
│   ├── detector.py
│   ├── segmenter.py
│   └── depth_estimator.py
├── pipeline/
│   ├── process_frame.py
│   ├── process_video.py
│   └── tracking.py
├── utils/
│   ├── visualization.py
│   ├── io.py
│   └── geometry.py
└── main.py
```

Each module encapsulates a single responsibility so that the pipeline can evolve as foundation models improve.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download model checkpoints**

   - `GroundingDino` weights (e.g. `groundingdino-swint-ogc`).
   - `SAM 2` checkpoint and configuration files from the official repository.
   - `Depth Anything v2` weights available on Hugging Face (`depth-anything/Depth-Anything-V2-*`).

   Update `configs/default.yaml` with the filesystem paths or Hugging Face identifiers.

3. **Run the pipeline**

   ```bash
   python main.py --config configs/default.yaml --input path/to/image_or_video
   ```

   Use `--mode image` or `--mode video` to control the processing path. For dataset-specific settings (e.g., KITTI intrinsics), create dedicated config files inheriting from `default.yaml`.

## License

This repository aggregates multiple third-party models with their respective licenses. Ensure compliance with each project's terms when deploying the pipeline.

