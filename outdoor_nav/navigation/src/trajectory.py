"""Trajectory generation and optimization"""
import sys
import os
import numpy as np
import math

#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from outdoor_nav.utils.utils import calculate_midpoints, pointcal
from outdoor_nav.planning.DF_FS_algorithm import (
    generate_cluster, generate_trajectories, detect_collisions,
    classify_and_filter_trajectories, choose_best_trajectory
)
from outdoor_nav.config import config


class TrajectoryPlanner:
    """Trajectory generation and planning"""
    
    def __init__(self):
        self.target_points = []
        self.angles = []
        self.angle_avg = 0
        self.stopflag = 0
    
    def calculate_midpath(self, left_edge, right_edge):
        """Calculate centerline between road edges"""
        return calculate_midpoints(left_edge, right_edge)
    
    def convert_to_world_coords(self, path):
        """Convert image coordinates to world coordinates"""
        if path is None:
            return None
        
        new_path = [pointcal(point) for point in path]
        new_path = sorted(new_path, key=lambda p: p[1])
        return new_path
    
    def generate_candidate_trajectories(self, path):
        """Generate clustered candidate trajectories"""
        C_path = generate_cluster(path)
        C_track = generate_trajectories(C_path, look_ahead=config.CLUSTER_LOOKAHEAD)
        return C_path, C_track
    
    def plan_safe_trajectory(self, path, trajectories, obstacles):
        """Plan safest trajectory avoiding obstacles"""
        if detect_collisions(path, obstacles):
            left_traj, right_traj, safe_traj = classify_and_filter_trajectories(
                trajectories, obstacles, path
            )
            
            if len(safe_traj) > 1:
                best = choose_best_trajectory(left_traj, right_traj)
                if len(best) > 7 and self.stopflag < config.MAX_STOPFLAG:
                    return best[6], best, True
                else:
                    self.stopflag += 1
                    return None, None, False
            else:
                self.stopflag += 1
                return None, None, False
        else:
            if len(path) > 3:
                return path[3], path, True
            return None, None, False
    
    def calculate_heading_angle(self, drive_point):
        """Calculate heading angle to target point"""
        if drive_point is None or drive_point == []:
            return None
        
        target_point = [drive_point[1] + 1.0, -drive_point[0]]
        angle = np.rad2deg(math.atan2(target_point[1], target_point[0]))
        return angle, target_point
    
    def smooth_angle(self, angle, angle_avg):
        """Smooth angle using buffer"""
        if abs(angle_avg - angle) < config.ANGLE_TOLERANCE:
            self.angles.append(angle)
            
            if len(self.angles) > config.PATH_SMOOTHING_WINDOW:
                self.angles = self.angles[-config.PATH_SMOOTHING_WINDOW:]
            
            return np.mean(self.angles), True
        return angle_avg, False
    
    def reset_stopflag(self):
        """Reset stop flag"""
        self.stopflag = 0


# Export for external use
__all__ = ['TrajectoryPlanner', 'pointcal']
