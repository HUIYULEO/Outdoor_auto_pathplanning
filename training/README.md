# Training Module

This folder contains scripts and utilities for training segmentation models. In this project, the models were trained by a dataset with 14000+ images about running road around DTU Campus and all the picture captured by Realsense D455 camera with Hopper robot.

## Directory Structure

```
training/
├─ README.md              # This file
├─ config.py             # Training configuration
├─ dataset.py            # Data loading and augmentation
├─ utils.py              # Training utilities
├─ train.py              # Main training script
└─ data_samples/         # (Optional) Sample data for testing 
    ├─ images/
    └─ masks/
```

## Dataset Format

Expected directory structure:

```
your_dataset/
├─ images/
│   ├─ img_001.png
│   ├─ img_002.png
│   └─ ...
└─ masks/
    ├─ img_001.png      (same filename as corresponding image)
    ├─ img_002.png
    └─ ...
```

- **images/**: RGB input images
- **masks/**: Grayscale segmentation masks (0 = background, 255 = road)

## Quick Start

### 1. Test Data Loading

```bash
python training/dataset.py
```

This will load sample data and display batch visualization.

### 2. Train Model

**Train UNet (recommended for real-time):**

```bash
python training/train.py \
    --data /path/to/dataset \
    --model_type unet \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3
```

**Train UNet++ (better accuracy):**

```bash
python training/train.py \
    --data /path/to/dataset \
    --model_type unetpp \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3
```

**Train Attention UNet:**

```bash
python training/train.py \
    --data /path/to/dataset \
    --model_type attunet \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3
```

## Training Parameters

| Parameter          | Default        | Description                              |
| ------------------ | -------------- | ---------------------------------------- |
| `--data`           | required       | Dataset root directory                   |
| `--model_type`     | unet           | Model architecture (unet/unetpp/attunet) |
| `--epochs`         | 50             | Number of training epochs                |
| `--batch_size`     | 8              | Batch size                               |
| `--lr`             | 1e-3           | Learning rate                            |
| `--checkpoint_dir` | ../checkpoints | Where to save models                     |
| `--device`         | cuda           | Device (cuda/cpu)                        |

## Output

- **checkpoints/**: Saved model weights
  - `{model_type}_epoch{N}.pth` - checkpoint at epoch N
  - `{model_type}_best.pth` - best validation loss model

## Data Augmentation

The dataset applies the following augmentations to training data:

- Random resize to (128, 128)
- ColorJitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet statistics

## Loss Function

Uses **BCEWithLogitsLoss** (Binary Cross-Entropy with Logits) for segmentation.

## Next Steps

After training, use the saved checkpoint in:

- `perception/src/segmenter.py` - for inference
- `main/main.py` - for real-time navigation
