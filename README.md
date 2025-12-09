# Outdoor Autonomous Path Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.9-blue)


End-to-end perception → planning → navigation system for outdoor mobile robots  
(Deployed on DTU Terrain Hopper Robot — MSc Thesis)

This project implements a complete autonomous navigation pipeline for an outdoor robot:
semantic segmentation, BEV projection, local path planning, and real-time control.
The same codebase was deployed on the DTU **Terrain Hopper** robot and tested in
real outdoor scenarios.

---

## Key Features
### **Perception**
- U-Net++ semantic segmentation for drivable-area detection, defined in `models/` and loaded through `perception/src/segmenter.py`  
- BEV (Bird’s Eye View) transformation & road-edge extraction  
- Supports RGB and RGB-D inputs  
- Lightweight inference wrapper for demos and deployment  

### **Navigation**
- Corridor-based / Midline Path planner for stable single-path following  
- Trajectory clustering + pure pursuit planning with collision checks (`planner/`, `navigation/src/trajectory.py`) 
- Safety checks: dead-end detection, short-path filtering, collision-band avoidance  

### **Robot Deployment**
- Integrated with DTU Terrain Hopper (`navigation/src/robot_interface.py`)
- Real-time loop with Intel RealSense D455 for perception → planning → control in `core/main.py`
- Training scripts for custom models in `training/` (see `training/README.md`)

---

## Repository Layout
- `outdoor_nav/` – main folder
  - `config/` – shared configuration
  - `utils/` – shared utilities
  - `perception/src/` – segmentation pipeline and BEV conversion helpers
  - `navigation/src/` – trajectory planner and robot client
  - `planning/` – clustered path generation, pure pursuit, and collision utilities
  - `models/` – UNet, UNet++, and Attention UNet definitions
  - `core/main.py` – real-time navigation loop (camera + robot required)
  - `perception/demos/` – run segmentation on a single RGB image
  - `navigation/demos/` – run the full planning stack on an offline mask
  - `checkpoint/` – sample pretrained weights (update paths in configs as needed)

- `training/` – dataset loader, training script, and README for training new checkpoints
## Quick Start

### Setup
1) Python 3.9+ recommended. CUDA GPU optional but helpful for real-time.
2) Install dependencies (adjust torch install for your platform):
```
pip install torch torchvision torchaudio
pip install opencv-python numpy pillow matplotlib torchvision torchsummary pyrealsense2
```
3) Verify the homography matrix at `utils/opt_homoMatrix.npy` matches your camera setup; replace it with your calibration if needed.

## Model Weights
- Configure the model path and type in `outdoor_nav/config.py` (`MODEL_PATH`, `MODEL_TYPE`, `DEVICE`).
- Sample checkpoints live under `checkpoint/` (e.g., `checkpoint/unet++/unet++_625.pth`). Point `MODEL_PATH` to the file you want to use.

## Demos

### Perception Demos
```
# Segmentation on a single image
python outdoor_nav/perception/demos/run_seg_on_image.py \
  --input outdoor_nav/perception/data_samples/example.png \
  --model checkpoint/unet++/unet++_625.pth \
  --threshold 0.5
```

### Offline Path planning（use test image）
```
python outdoor_nav/navigation/demos/run_full_nav_offline.py \
  --input path/to/image_folder \
  --model checkpoint/unet++/unet++_625.pth \
  --threshold 0.5
```
Pass a binary drivable-area mask (same size as your BEV space). The script projects to BEV, extracts edges, clusters trajectories, filters collisions, and visualizes the best path

### Full navigation system

This command starts the **full navigation stack** that was deployed on the DTU Terrain Hopper robot, connecting perception, planning and low-level Mobotware commands. 

```
python -m outdoor_nav.core.navigation_system
```
- Expects a RealSense camera (see `src/imagecapture.py`).
- Robot IP/port and motion parameters come from `config/config.py`.
- The loop: capture RGB -> segment -> BEV -> edge extraction -> trajectory generation/filtering -> send drive commands.


## Training Your Own Model
- See `training/README.md` for details. Typical command:
```
python training/train.py --data /path/to/dataset --model_type unetpp --epochs 50 --batch_size 8
```
- Dataset format: `images/` and `masks/` folders with matching filenames; masks are binary (0/255).

## Notes and Tips
- If you change the camera or mounting height, recalibrate the homography (`src/opt_homoMatrix.npy`).
- For smoother steering, tune `ANGLE_TOLERANCE`, `ANGLE_THRESHOLD`, `PATH_SMOOTHING_WINDOW`, and `MAX_STOPFLAG` in `core/config.py`.
- Offline demos avoid robot/camera dependencies; use them to validate new models or homographies before field tests.
- Use relative paths where possible; scripts resolve to project root if a path is not absolute.

## Project structure

```
Outdoor_auto_pathplanning
│  .gitignore
│  LICENSE
│  README.md                     <- The top-level README for developers using this project.
│  requirements.txt              <- The requirements file for reproducing the analysis environment
│
├─outdoor_nav                    <- Main source code for use in this project
│  │  __init__.py                <- Makes src a Python module
│  │
│  ├─checkpoint                  <- Location where trained models are saved. Contains final model "unet++_625.torch"
│  │  ├─unet
│  │  │      unet_ND618.pth
│  │  │
│  │  └─unet++
│  │          unet++16_150.pth
│  │          unet++_625.pth
│  │
│  ├─config
│  │  │  config.py
│  │  │  __init__.py
│  │
│  ├─core
│  │  │  main.py                  <- Main code for running the full navigation stack that was deployed on the DTU Terrain Hopper robot.
│  │  │  __init__.py
│  │
│  ├─models                       <- Models defination
│  │  │  attunet.py
│  │  │  unet.py
│  │  │  unetpp.py
│  │  │  __init__.py
│  │
│  ├─navigation                   <- Navigation module
│  │  │  README.md
│  │  │
│  │  ├─data_samples
│  │  ├─demos
│  │  │      run_full_nav_offline.py
│  │  │
│  │  └─src
│  │      │  robot_interface.py
│  │      │  trajectory.py
│  │      └─ __init__.py
│  │
│  ├─perception                   <- Perception module
│  │  │  README.md
│  │  │
│  │  ├─data_samples              <- Location for pictures used in demos.
│  │  │      example.png
│  │  │      true_mask.png
│  │  │
│  │  ├─demos
│  │  │      run_seg_on_image.py
│  │  │
│  │  └─src
│  │      │  bev_transform.py
│  │      │  imagecapture.py
│  │      │  segmenter.py
│  │      └─ __init__.py
│  │
│  ├─planning                   <- path planning algrithom
│  │  │  AStar.py
│  │  │  DF_FS_algorithm.py
│  │  │  pure_planner.py
│  │  │  SplitLR_test.py
│  │  └─ __init__.py
│  │  
│  │
│  └─utils
│     │  opt_homoMatrix.npy
│     │  utils.py                 <- Functions to control robot and devices to execute the perception and navigation
│     └─ __init__.py
│
│
└─training
        config.py
        dataset.py            <- Scripts to generate data and create the readers for the training and testing data
        README.md
        train.py              <- Main file to train and evaluate models and run inferences on images.
        utils.py
```

## License
This project is licensed under the terms of the MIT license.

