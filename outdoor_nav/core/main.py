"""Main real-time navigation loop with restructured modules"""
import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from outdoor_nav.config import config
from outdoor_nav.perception.src.imagecapture import ImageCaptureAsync
from outdoor_nav.perception.src.segmenter import RoadSegmenter
from outdoor_nav.perception.src.bev_transform import BEVTransformer
from outdoor_nav.navigation.src.robot_interface import RobotController
from outdoor_nav.navigation.src.trajectory import TrajectoryPlanner


class NavigationSystem:
    """Integrated robot navigation system"""
    
    def __init__(self):
        self.robot = None
        self.segmenter = None
        self.camera = None
        self.planner = None
        self.bev_transformer = BEVTransformer()
    
    def initialize(self):
        """Initialize all system components"""
        print("=" * 60 + "\n")
        print("Robot Navigation System Initialization")
                
        # Initialize robot
        print("\n[1/4] Initializing robot connection...")
        self.robot = RobotController(
            config.ROBOT_IP, 
            config.ROBOT_PORT, 
            config.CONNECTION_TIMEOUT
        )
        self.robot.initialize_odometry()
        odox, odoy, odoth = self.robot.get_odometry()
        print(f' Odometry initialized: x={odox:.3f}, y={odoy:.3f}, θ={np.rad2deg(odoth):.3f}°')
        
        # Initialize segmentation model
        print("\n[2/4] Initializing segmentation model...")
        self.segmenter = RoadSegmenter(
            model_path=config.MODEL_PATH,
            model_type=config.MODEL_TYPE
        )
        print(f' Model loaded: {config.MODEL_TYPE} ({config.MODEL_PATH})')
        
        # Initialize camera
        print("\n[3/4] Initializing camera...")
        self.camera = ImageCaptureAsync(0)
        self.camera.start()
        print(' Camera started')
        
        # Initialize trajectory planner
        print("\n[4/4] Initializing trajectory planner...")
        self.planner = TrajectoryPlanner()
        self.planner.angle_avg = np.rad2deg(odoth)
        print('  Trajectory planner ready')
        
        print("System initialization complete! Starting navigation loop...")
        print("=" * 60 + "\n")
    
    def run(self):
        """Main navigation loop"""
        try:
            frame_count = 0
            while True:
                frame_count += 1
                
                # Get current odometry
                odox, odoy, odoth = self.robot.get_odometry()
                
                # Capture and process frame
                ret, color_image = self.camera.read()
                if not ret:
                    continue
                
                color_image = cv.resize(
                    color_image, 
                    config.CAMERA_OUTPUT_SIZE,
                    interpolation=cv.INTER_AREA
                )
                
                # Segment road using deep learning
                mask, contours = self.segmenter.segment_and_clean(color_image)
                
                # Convert to bird's eye view
                bev_color = self.bev_transformer.transform(color_image)
                bev_mask = self.bev_transformer.transform(mask)
                
                # Display segmentation mask
                cv.namedWindow('Segmentation Mask', cv.WINDOW_AUTOSIZE)
                cv.imshow("Segmentation Mask", cv.resize(mask, (640, 480)))
                
                # Extract road edges from BEV
                (largest_contour, left_edge, right_edge, 
                 bottom_center) = self.bev_transformer.extract_road_edges(bev_mask)
                
                # Check if road is detected
                if largest_contour is None:
                    print(f"[Frame {frame_count}] No road detected!")
                    self.robot.stop()
                    time.sleep(0.1)
                    self.planner.stopflag = 4
                    time.sleep(1)
                    continue
                
                # Calculate centerline path
                path = self.planner.calculate_midpath(left_edge, right_edge)
                new_path = self.planner.convert_to_world_coords(path)
                
                if new_path is None or len(new_path) <= 4:
                    print(f"[Frame {frame_count}] Path too short!")
                    self.robot.stop()
                    self.planner.stopflag = 4
                    continue
                
                # Generate candidate trajectories
                C_path, C_track = self.planner.generate_candidate_trajectories(new_path)
                
                # Extract obstacle positions
                obstacles = self._extract_obstacles(largest_contour, left_edge, right_edge)
                
                # Plan safe trajectory
                (drive_point, best_trajectory, 
                 success) = self.planner.plan_safe_trajectory(new_path, C_track, obstacles)
                
                if not success:
                    print(f"[Frame {frame_count}] Cannot find safe trajectory "
                          f"(stopflag={self.planner.stopflag})")
                    if self.planner.stopflag > config.MAX_STOPFLAG:
                        self.robot.drive_with_timeout(
                            odox, odoy, 
                            np.rad2deg(odoth), 
                            config.VELOCITY
                        )
                    continue
                
                # Calculate heading angle
                angle_result = self.planner.calculate_heading_angle(drive_point)
                if angle_result is None:
                    continue
                
                angle, target_point = angle_result
                
                # Smooth angle using buffer
                angle_avg, is_valid = self.planner.smooth_angle(angle, self.planner.angle_avg)
                
                if is_valid and len(self.planner.angles) >= config.PATH_SMOOTHING_WINDOW:
                    angle_avg = self.planner.angle_avg
                    
                    # Determine motion command
                    if (0 < abs(angle_avg) < config.ANGLE_THRESHOLD and 
                        self.planner.stopflag < config.MAX_STOPFLAG):
                        th = np.rad2deg(odoth) + angle_avg
                        self.robot.drive_to(odox, odoy, th, config.VELOCITY)
                        print(f"  → Drive to θ={th:.2f}° @ v={config.VELOCITY}")
                        self.planner.reset_stopflag()
                    elif self.planner.stopflag < config.MAX_STOPFLAG:
                        th = np.rad2deg(odoth)
                        self.robot.drive_with_timeout(odox, odoy, th, config.VELOCITY)
                        self.planner.reset_stopflag()
                
                # Visualization
                self._visualize_planning(C_path, obstacles, best_trajectory, 
                                        left_edge, right_edge)
                
                cv.namedWindow('BEV Result', cv.WINDOW_AUTOSIZE)
                cv.imshow("BEV Result", cv.resize(
                    bev_color, 
                    (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
                ))
                
                # Check for exit command
                key = cv.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("\nQuit command received")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def _extract_obstacles(self, largest_contour, l_edge, r_edge):
        """Extract obstacle positions from contour and edges"""
        from navigation.src.trajectory import pointcal
        
        obstacles = []
        
        # Add contour points as obstacles
        for obs in largest_contour:
            obs_pos = pointcal(obs[0])
            obstacles.append(obs_pos)
        
        # Add road edges as obstacles
        obstacles.extend(l_edge)
        obstacles.extend(r_edge)
        
        return obstacles
    
    def _visualize_planning(self, C_path, obstacles, best_trajectory, left_edge, right_edge):
        """Visualize path planning results"""
        plt.cla()
        
        # Plot clustered paths
        for path_item in C_path:
            X, Y = zip(*path_item)
            plt.plot(X, Y, alpha=0.6)
        
        # Plot obstacles
        for x_obs, y_obs in obstacles:
            circle = plt.Circle((x_obs, y_obs), 0.3, color='r', fill=True, alpha=0.7)
            plt.gcf().gca().add_artist(circle)
        
        # Plot best trajectory
        if best_trajectory:
            for point in best_trajectory:
                x, y = point
                plt.plot(x, y, ".g", markersize=4)
        
        # Plot road edges
        left_edge = np.array(left_edge)
        right_edge = np.array(right_edge)
        plt.plot(left_edge[:, 0], left_edge[:, 1], "ok", markersize=3, label='Left edge')
        plt.plot(right_edge[:, 0], right_edge[:, 1], "ok", markersize=3, label='Right edge')
        
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=8)
        plt.pause(0.0001)
    
    def shutdown(self):
        """Cleanup and shutdown"""
        print("\nShutting down...")
        cv.destroyAllWindows()
        
        if self.camera is not None:
            print("  → Stopping camera...")
            self.camera.stop()
        
        if self.robot is not None:
            print("  → Disconnecting robot...")
            self.robot.disconnect()
        
        print("Shutdown complete.\n")


def main():
    """Main entry point"""
    nav_system = NavigationSystem()
    nav_system.initialize()
    nav_system.run()


if __name__ == "__main__":
    main()
