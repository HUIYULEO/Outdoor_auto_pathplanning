"""Demo: Segmentation on single image"""
import sys
import os
import argparse
import cv2 as cv
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from outdoor_nav.perception.src.segmenter import RoadSegmenter
from outdoor_nav.config import config


def main():
    parser = argparse.ArgumentParser(description='Road segmentation demo')
    
    # 设置默认输入路径（使用绝对路径）
    default_input = project_root / 'outdoor_nav' / 'perception' / 'data_samples' / 'example.png'
    parser.add_argument('--input', type=str, default=str(default_input),
                       help='Input image path')
    
    # 处理模型路径
    model_path = config.MODEL_PATH
    if isinstance(model_path, Path):
        if not model_path.is_absolute():
            model_path = project_root / model_path
    elif isinstance(model_path, str):
        if not os.path.isabs(model_path):
            model_path = project_root / model_path
    
    # 如果默认模型不存在，尝试其他模型
    if not Path(model_path).exists():
        alternative_models = [
            project_root / 'checkpoint' / 'unet++' / 'unet++_625.pth',
            project_root / 'checkpoint' / 'unet++' / 'unet++_618.pth',
            project_root / 'checkpoint' / 'unet++' / 'unet++16_150.pth',
        ]
        for alt_model in alternative_models:
            if alt_model.exists():
                print(f"Default model not found, using: {alt_model}")
                model_path = alt_model
                break
    
    parser.add_argument('--model', type=str, default=str(model_path),
                       help='Model checkpoint path')
    
    parser.add_argument('--output', type=str, default='output_mask.png',
                       help='Output mask path')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Segmentation threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # 处理输入路径
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / args.input
        if not input_path.exists():
            input_path = project_root / 'outdoor_nav' / args.input
    
    # 加载图像 - 完全按照 final_nav_test.py
    print(f"Loading image: {input_path}")
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return
    
    color_image = cv.imread(str(input_path))
    if color_image is None:
        print(f"Error: Cannot load image from {input_path}")
        return
    
    print(f"Image shape: {color_image.shape}")
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / args.model
    
    print(f"\nModel path: {model_path}")
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Please check the model path in config.py or use --model argument")
        return
    else:
        print(f"✓ Model file found: {model_path}")
    
    # 初始化分割器
    print(f"\nInitializing segmenter with model type: {config.MODEL_TYPE}")
    try:
        segmenter = RoadSegmenter(model_path=str(model_path), 
                                 model_type=config.MODEL_TYPE)
        print("✓ Segmenter initialized successfully")
    except Exception as e:
        print(f"ERROR initializing segmenter: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 执行分割 
    print(f"\nSegmenting with threshold: {args.threshold}")
    try:
        mask, contours = segmenter.segment_and_clean(color_image, threshold=args.threshold)
        if mask.max() == 0:
            print("\n WARNING: Mask is completely black (no segmentation detected)")
    except Exception as e:
        print(f"ERROR during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 生成叠加图像 
    overlay_color = (150, 150, 150)  
    result = color_image.copy()
    result = cv.resize(result, (128, 128))
    result[np.where(mask)] = overlay_color
    
    # 保存结果
    cv.imwrite(args.output, mask)
    print(f"\n✓ Mask saved to: {args.output}")
    
    # 显示结果
    print("\nDisplaying results...")
    cv.namedWindow('Original', cv.WINDOW_NORMAL)
    cv.namedWindow('Mask', cv.WINDOW_NORMAL)
    cv.namedWindow('Overlay', cv.WINDOW_NORMAL)
    
    # 显示原始图像（调整大小以便查看）
    display_size = (640, 480)
    cv.imshow('Original', cv.resize(color_image, display_size))
    cv.imshow('Mask', cv.resize(mask, display_size))
    cv.imshow('Overlay', cv.resize(result, display_size))
    
    print("Press any key to exit...")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()