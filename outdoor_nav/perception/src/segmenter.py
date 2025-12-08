"""Unified road segmentation interface"""
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from outdoor_nav.models.unet import UNet
from outdoor_nav.models.unetpp import NestedUNet
from outdoor_nav.models.attunet import AttU_Net
from outdoor_nav.config import config
from outdoor_nav.utils.utils import removeSmallAreas, mergeContours


class RoadSegmenter:
    """Road segmentation module using deep learning
    
    Supports multiple model architectures:
    - 'unet': Standard UNet
    - 'unetpp': UNet++ (NestedUNet)
    - 'attunet': Attention-based UNet
    """
    
    def __init__(self, model_path=None, model_type='unet', device=None):
        """
        Initialize segmenter with specified model
        
        Args:
            model_path: path to pretrained weights
            model_type: 'unet', 'unetpp', or 'attunet'
            device: torch device (cuda or cpu)
        """
        self.device = device or torch.device(
            config.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        self.model_type = model_type
        
        # Load model architecture
        self._load_model(model_path)
        
        # Preprocessing pipeline - 完全按照 final_nav_test.py
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load model architecture and pretrained weights"""
        print(f"Loading {self.model_type.upper()} model...")
        
        # Create model instance based on type
        if self.model_type == 'unet':
            self.model = UNet().to(self.device)
        elif self.model_type == 'unetpp':
            self.model = NestedUNet().to(self.device)
        elif self.model_type == 'attunet':
            self.model = AttU_Net().to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           "Choose from: 'unet', 'unetpp', 'attunet'")
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"✓ Model ready on {self.device}")
    
    def segment(self, color_image, threshold=None):
        """
        Generate segmentation mask from color image
        完全按照 final_nav_test.py 的实现
        
        Args:
            color_image: numpy array (H, W, 3) in BGR format (from cv.imread)
            threshold: mask threshold (default: config.MASK_THRESHOLD)
        
        Returns:
            mask: binary mask (H, W) in [0, 255]
        """
        threshold = threshold or config.MASK_THRESHOLD
        
        # 完全按照 final_nav_test.py 的预处理方式
        # 注意：没有 BGR 到 RGB 的转换！
        pil_image = Image.fromarray(color_image)
        preprocessed = self.transform(pil_image)
        
        # 添加 batch 维度：从 (C, H, W) 变为 (1, C, H, W)
        input_tensor = preprocessed.unsqueeze(0).to(self.device)
        
        # Inference - 完全按照 final_nav_test.py
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # 处理输出 - 完全按照 final_nav_test.py
        pred_img = F.sigmoid(outputs)
        pred_img.detach().numpy()  # 虽然不赋值，但保持一致性
        
        # 创建二值化 mask - 完全按照 final_nav_test.py 的方式
        mask = (pred_img > threshold).float()  # Binarize the prediction
        mask = mask.squeeze(0).squeeze(0).numpy()
        
        mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
        mask[mask > 0.5] = 255
        mask[mask < 0.5] = 0
        
        return mask
    
    def segment_and_clean(self, color_image, threshold=None):
        """
        Segment + remove small areas + merge contours
        完全按照 final_nav_test.py 的处理流程
        
        Returns:
            mask: cleaned binary mask
            contours: list of contours
        """
        mask = self.segment(color_image, threshold)
        
        # 使用 removeSmallAreas - 完全按照 final_nav_test.py
        mask, contours = removeSmallAreas(mask, config.MIN_AREA_SIZE)
        
        if len(contours) > 0:
            # Generate approximate contour to reduce overall amount of data
            # 完全按照 final_nav_test.py
            for i in range(len(contours)):
                contours[i] = cv.approxPolyDP(contours[i], 2, True)
            
            # If exactly two contours remain in the image, try to merge them
            if len(contours) == 2:
                mask, contours = mergeContours(mask, contours)
                if len(contours) > 0 and len(contours[0]) > 4:
                    mask = cv.drawContours(mask, contours, 0, (255, 255, 255), cv.FILLED)
        
        return mask, contours
    
    def overlay_mask(self, color_image, mask, alpha=0.5, color=(150, 150, 150)):
        """Overlay mask on original image - 完全按照 final_nav_test.py"""
        result = color_image.copy()
        result = cv.resize(result, (128, 128))
        result[np.where(mask)] = color
        return result