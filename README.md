# Outdoor Autonomous Path Planning
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

A modular pipeline for outdoor robot navigation with segmentation, BEV transform, trajectory generation, and collision-aware planning.

## Features
- Road segmentation (UNet / UNet++ / Attention UNet)
- Bird’s Eye View (BEV) transform
- Midline extraction, clustering, trajectory generation
- Collision detection and best-trajectory selection

## Project structure

```
outdoor_nav/
├── models/ # Deep learning model
├── perception/ # Perception Module（segmentation for road）
├── planning/ # Path Planning Module
├── navigation/ # Navigation Control Module
└── utils/ # Utility  function
```

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

```
#used for DTU Terrain Hopper robot
python -m outdoor_nav.core.navigation_system
```

## Training Model

Check [training/README.md](training/README.md)

## Configureation

Check outdoor_nav/config/config.py

## Notes

- Large checkpoints (`checkpoint/*.pth`) are not committed; place them locally or provide download links.
- Use relative paths where possible; scripts resolve to project root if a path is not absolute.
- For GPU inference, set `USE_CUDA=true` and ensure CUDA is available.

## License
MIT (see `LICENSE`).