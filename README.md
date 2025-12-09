# Outdoor Autonomous Path Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NVIDIA Flex](https://img.shields.io/badge/NVIDIA-Flex-green)](https://developer.nvidia.com/flex)
![Python](https://img.shields.io/badge/Python-3.9-blue)


The project implements an end-to-end **perception-to-navigation** pipeline for an
outdoor mobile robot (Terrain Hopper):

- **Perception module** – semantic segmentation of drivable area and BEV
  transformation from RGB(-D) images.
- **Navigation module** – local path planning (A*/corridor/MLP-style planners),
  Pure Pursuit motion control, and example scripts used on the real robot system
  (Hopper robot test).

## Features
- Road segmentation (UNet / UNet++ / Attention UNet)
- Bird’s Eye View (BEV) transform
- Midline extraction, clustering, trajectory generation
- Collision detection and best-trajectory selection

## Quick Start

### Installation

```
pip install -r requirements.txt
```
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


### Full navigation system

This command starts the **full navigation stack** that was deployed on the DTU Terrain Hopper robot, connecting perception, planning and low-level
Mobotware commands.
```
python -m outdoor_nav.core.navigation_system
```

## Training Model

Check [training/README.md](training/README.md)

## Configuration

Check outdoor_nav/config/config.py

## Notes

- Large checkpoints (`checkpoint/*.pth`) are not committed; place them locally or provide download links.
- Use relative paths where possible; scripts resolve to project root if a path is not absolute.
- For GPU inference, set `USE_CUDA=true` and ensure CUDA is available.


## Project structure

```
Outdoor_auto_pathplanning
│  .gitignore
│  LICENSE
│  README.md
│  requirements.txt
│
├─outdoor_nav
│  │  __init__.py
│  │
│  ├─checkpoint
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
│  │  │  main.py
│  │  │  __init__.py
│  │
│  ├─models
│  │  │  attunet.py
│  │  │  unet.py
│  │  │  unetpp.py
│  │  │  __init__.py
│  │
│  ├─navigation
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
│  ├─perception
│  │  │  README.md
│  │  │
│  │  ├─data_samples
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
│  ├─planning
│  │  │  AStar.py
│  │  │  DF_FS_algorithm.py
│  │  │  DWA.py
│  │  │  JPS.py
│  │  │  pure_planner.py
│  │  │  SplitLR_test.py
│  │  └─ __init__.py
│  │  
│  │
│  └─utils
│     │  opt_homoMatrix.npy
│     │  utils.py
│     └─ __init__.py
│
│
└─training
        config.py
        dataset.py
        README.md
        train.py
        utils.py
```

## License
This project is licensed under the terms of the MIT license.

