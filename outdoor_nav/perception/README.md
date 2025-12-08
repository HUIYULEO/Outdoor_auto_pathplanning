# Perception Module

Semantic segmentation of drivable areas for outdoor navigation, plus basic
BEV / coordinate utilities.

- `src/segmenter.py` – wraps a U-Net / U-Net++ model for drivable-area segmentation
- `src/bev_transform.py` – homography / BEV projection helpers
- `src/imagecapture.py` – image capture function for realsense camera
- `demos/run_seg_on_image.py` – small CLI demo that runs segmentation on a
  sample image and saves an overlay
