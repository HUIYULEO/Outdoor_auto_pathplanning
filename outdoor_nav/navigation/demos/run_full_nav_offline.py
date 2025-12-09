"""Demo: Full navigation pipeline on offline image"""
import sys
import os
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from outdoor_nav.perception.src.segmenter import RoadSegmenter
from outdoor_nav.perception.src.bev_transform import BEVTransformer
from outdoor_nav.navigation.src.trajectory import TrajectoryPlanner
from outdoor_nav.config import config
from outdoor_nav.utils.utils import (
    img2BEV, test_find_edges, findLowerCorners, 
    calculate_midpoints, pointcal, coordinateTrans, goalSearch,
    removeSmallAreas, mergeContours, removeSmallAreas_compat
)
from outdoor_nav.planning.DF_FS_algorithm import (
    generate_cluster, generate_trajectories, detect_collisions,
    classify_and_filter_trajectories, choose_best_trajectory
)


# Define a custom dataset
class ImageDataset(Dataset):
    def __init__(self, FOLDER_PATH):
        self.image_paths = [os.path.join(FOLDER_PATH, file) for file in os.listdir(FOLDER_PATH)
                            if file.endswith(".jpg") or file.endswith(".png")]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv.imread(image_path)
        test_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Preprocess the color image
        pil_image = Image.fromarray(image)
        image_tensor = test_data_transform(pil_image)
        return image_tensor, image_path

    def __len__(self):
        return len(self.image_paths)



def main():
    parser = argparse.ArgumentParser(description='Full offline navigation demo')
    parser.add_argument('--input', type=str, default=str(config.DATA_DIR),
                        help='Input image folder or single image')
    parser.add_argument('--model', type=str, default=str(config.MODEL_PATH),
                        help='Model checkpoint path')
    parser.add_argument('--threshold', type=float, default=config.MASK_THRESHOLD,
                        help='Segmentation threshold')
    parser.add_argument('--output', type=str, default='nav_result.png',
                        help='Output visualization')
    args = parser.parse_args()

    
    # Check model path
    model_path = config.MODEL_PATH
    print(model_path)
    if isinstance(model_path, Path):
        if not model_path.is_absolute():
            model_path = project_root / model_path
    elif isinstance(model_path, str):
        if not os.path.isabs(model_path):
            model_path = project_root / model_path
        
    # segmenter init
    print(f"\nInitializing segmenter with model type: {config.MODEL_TYPE}")
    try:
        segmenter = RoadSegmenter(model_path=str(model_path), 
                                 model_type=config.MODEL_TYPE)
        print("Segmenter initialized successfully")
    except Exception as e:
        print(f"ERROR initializing segmenter: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Dataloader
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / args.input
    
    print(f"\nLoading dataset from: {input_path}")
    if not input_path.exists():
        print(f"Error: Path not found at {input_path}")
        return
    
    # Detect folder vs single image
    if input_path.is_file():
        # Single image: create a temporary dataset
        dataset = ImageDataset(str(input_path.parent))
        image_name = input_path.name
        dataset.image_paths = [str(input_path)]
    else:
        # Folder
        dataset = ImageDataset(str(input_path))
    
    if len(dataset) == 0:
        print("Error: No images found in the specified path")
        return
        
    # Create data loader 
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Iterate through images 
    for images, image_paths in data_loader:        
        # Load original image
        original_image = cv.imread(image_paths[0])
        if original_image is None:
            print(f"Error: Cannot load image from {image_paths[0]}")
            continue
        
        # Segmentation
        try:
            mask, contours = segmenter.segment_and_clean(original_image, threshold=args.threshold)
            
            if mask.max() == 0:
                print("WARNING: Mask is completely black (no segmentation detected)")
                continue
        except Exception as e:
            print(f"ERROR during segmentation: {e}")
            continue
        
        # Build overlay
        overlay_color = (150, 150, 150)
        result = original_image.copy()
        result = cv.resize(result, (128, 128))
        result[np.where(mask)] = overlay_color
        
        # Use compatibility wrapper
        mask, contours = removeSmallAreas_compat(mask, 120)
        
        if len(contours) > 0:
            # Generate approximate contour to reduce overall amount of data
            for i in range(len(contours)):
                contours[i] = cv.approxPolyDP(contours[i], 2, True)
            
            # If exactly two contours remain in the image, try to merge them
            if len(contours) == 2:
                mask, contours = mergeContours(mask, contours)
                if len(contours) > 0 and len(contours[0]) > 4:
                    mask = cv.drawContours(mask, contours, 0, (255, 255, 255), cv.FILLED)
        
        # BEV
        result = img2BEV(result)
        warpedmask = img2BEV(mask)
        
        # Extract road edges
        findContours_result = cv.findContours(warpedmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(findContours_result) == 3:
            _, warpcontours, _ = findContours_result
        else:
            warpcontours, _ = findContours_result
        
        warpcontours = [cv.approxPolyDP(cnt, 2, True) for cnt in warpcontours]
        
        if len(warpcontours) == 0:
            print("Error: No road detected in BEV mask!")
            continue
        
        # Compute key points and edges
        largest_contour = max(warpcontours, key=cv.contourArea)
        lower_left, lower_right = findLowerCorners(largest_contour)
        bottom_center = (lower_left[0] + lower_right[0]) // 2, (lower_left[1] + lower_right[1]) // 2
        
        left_edge, right_edge = test_find_edges(warpedmask)
        left_edge.append(lower_left)
        right_edge.append(lower_right)
        
        # Visualize edge points
        height, width = result.shape[0], result.shape[1]
        for left_point in left_edge:
            cv.circle(result, left_point, 2, (255, 255, 0), -1)
        
        for right_point in right_edge:
            cv.circle(result, right_point, 2, (255, 0, 255), -1)
        
        for point in largest_contour:
            for obs in point:
                x, y = obs
                cv.circle(result, (x, y), 4, (0, 0, 0), -1)
        
        # Compute goal and path
        top_center = goalSearch(left_edge, right_edge, bottom_center)
        cv.circle(result, bottom_center, 4, (0, 255, 255), -1)
        
        path = calculate_midpoints(left_edge, right_edge)
        
        # Draw path
        drive_point = [0.5, 0]
        if path is not None:
            for point in path:
                x, y = point
                if 0 <= y < height and 0 <= x < width:
                    result[y, x] = [255, 0, 0]
            mid_index = len(path) // 2 if len(path) % 2 == 1 else (len(path) + 1) // 2
            if len(path) > 4:
                drive_point = path[mid_index]
                cv.arrowedLine(result, path[mid_index + 1], drive_point, (0, 0, 255), 1, 0, 0, 0.2)
            else:
                drive_point = [0.5, 0]
        
        # Coordinate transform
        new_path = []
        if path is not None:
            for point in path:
                point = pointcal(point)
                new_path.append(point)
            mid_index = len(new_path) // 2 if len(new_path) % 2 == 1 else (len(new_path) + 1) // 2
            drive_point = new_path[-mid_index]
        
        if len(left_edge) > 4 and len(right_edge) > 4:
            start, end, l_edge, r_edge = coordinateTrans(bottom_center, left_edge, right_edge)
        else:
            print('Stop: road edges too short')
            continue
        
        if top_center is not None:
            end = pointcal(top_center)
        
        # Generate trajectories
        C_path = generate_cluster(new_path)
        C_track = generate_trajectories(C_path, look_ahead=1.5)
        
        # Collect obstacles
        obstacles = []
        for obs in largest_contour:
            obs_pos = pointcal(obs[0])
            obstacles.append(obs_pos)
        for left_point in l_edge:
            obstacles.append(left_point)
        for right_point in r_edge:
            obstacles.append(right_point)
        
        # Collision check and trajectory selection
        if detect_collisions(new_path, obstacles):
            left_trajectories, right_trajectories, safe_trajectories = classify_and_filter_trajectories(
                C_track, obstacles, new_path
            )
            if len(safe_trajectories) != 0:
                best_trajectory = choose_best_trajectory(left_trajectories, right_trajectories)
            else:
                print('Stop: no safe trajectory')
                continue
        else:
            left_trajectories, right_trajectories, safe_trajectories = C_track, C_track, C_track
            best_trajectory = new_path
        
        # Visualization
        show_plt = False
        cv.namedWindow('Original Image vs Segmentation', cv.WINDOW_AUTOSIZE)
        cv.imshow("Original Image vs Segmentation", cv.resize(result, (600, 600)))
        
        plt.cla()
        
        for x_obs, y_obs in obstacles:
            radius = 0.2
            circle = plt.Circle((x_obs, y_obs), radius, color='r', fill=True)
            plt.gcf().gca().add_artist(circle)
        
        
        for track in C_track:
            for point in track:
                x, y = point
                plt.plot(x, y, ".b", label="trajectory")
        
        for track in safe_trajectories:
            for point in track:
                xs, ys = point
                plt.plot(xs, ys, ".y")
        
        for point in best_trajectory:
            xb, yb = point
            plt.plot(xb, yb, "og")
        
        # Plot road edges
        l_edge = np.array(l_edge)
        r_edge = np.array(r_edge)
        plt.plot(l_edge[:, 0], l_edge[:, 1], "ok")
        plt.plot(r_edge[:, 0], r_edge[:, 1], "ok")
        
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
        
        key = cv.waitKey(0)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            break
    
    print("\n All images processed!")


if __name__ == '__main__':
    main()