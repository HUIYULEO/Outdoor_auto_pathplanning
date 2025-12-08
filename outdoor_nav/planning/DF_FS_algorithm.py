import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from outdoor_nav.planning.pure_planner import *

def generate_cluster(ref_path, num_paths=7, delta=0.2):
    """
    Generate a cluster of paths by translating the reference path left and right by a delta distance.

    Parameters:
    ref_path (list of tuples): The reference path, each tuple contains x, y coordinates of a waypoint.
    num_paths (int): The number of paths to generate in the cluster, including the reference path. Must be odd.
    delta (float): The distance to translate the path for each step.

    Returns:
    C_path (list of lists): The cluster of paths. Each path is a list of tuples containing x, y coordinates.
    """

    assert num_paths % 2 == 1, "Number of paths must be odd."

    C_path = []
    mid_path_index = num_paths // 2

    for i in range(num_paths):
        new_path = []
        for waypoint in ref_path:
            x, y = waypoint
            translated_x = x + (i - mid_path_index) * delta
            new_path.append((translated_x, y))
        C_path.append(new_path)

    return C_path


def generate_trajectories(C_path, look_ahead, v=1.0, dt=0.2, L=1.0):
    """
    Apply the Pure Pursuit algorithm to each path in the cluster, generating a predicted trajectory for each path.
    """
    C_track = []
    for path in C_path:
        track = pure_pursuit(path, look_ahead, v=v, dt=dt, L=L)
        # Add the predicted trajectory to the cluster of trajectories
        C_track.append(track)
    return C_track

def detect_collisions(track, obstacles):
    """
    Check if a trajectory collides with any obstacles.
    Returns True if there is a collision, False otherwise.
    """
    radius = 0.25
    for x, y in track:
        for x_obs, y_obs in obstacles:
            if np.hypot(x - x_obs, y - y_obs) < radius:
                # A collision has been detected
                return True
    # No collisions were detected
    return False

def classify_and_filter_trajectories(C_track, obstacles, ref_path):
    """
    Classify each trajectory as left or right of the reference, and filter out any trajectories that collide with obstacles.
    """
    left_trajectories = []
    right_trajectories = []
    safe_trajectories = []

    for track in C_track:
        # Check for collisions
        if detect_collisions(track, obstacles):
            continue  # Skip this trajectory if it collides with an obstacle

        safe_trajectories.append(track)

        # Classify as left or right based on the final position
        final_x, _ = track[-1]
        if final_x < ref_path[-1][0]:
            left_trajectories.append(track)
        else:
            right_trajectories.append(track)
    print() 
    return left_trajectories, right_trajectories, safe_trajectories

def choose_best_trajectory(left_trajectories, right_trajectories):
    """
    Compare the two trajectory sets (left and right),
    choose the one with more trajectories,
    and return the center trajectory of the trajectory set as the best choice.
    """
    # Check which trajectory set has more trajectories
    if len(left_trajectories) > len(right_trajectories):
        chosen_trajectories = left_trajectories
    else:
        chosen_trajectories = right_trajectories
    # print(len(ref_path))
    # Calculate the index of the center trajectory
    if len(chosen_trajectories) > 2:
        center_index = len(chosen_trajectories) // 2
        return chosen_trajectories[center_index]
    else:
        return chosen_trajectories[0]



def main():
    return None


if __name__ == '__main__':
    # Example usage
    ref_path = [(0, 0), (0.3, 1), (0.5, 2), (0.7, 3), (0.8, 4), (1.0,5), (1.0,6), (1.0, 7), (1.0, 8)]

    C_path = generate_cluster(ref_path)

    # Example usage
    C_track = generate_trajectories(C_path, look_ahead=1.0)

    obstacles = [(0.7, 2.5), (0.9, 2.5)]

    left_trajectories, right_trajectories, safe_trajectories = classify_and_filter_trajectories(C_track, obstacles,ref_path)

    # Example usage:
    best_trajectory = choose_best_trajectory(left_trajectories, right_trajectories)

    # Print all paths in the cluster for inspection
    for path in C_path:
        print(path)

    # Prepare the figure
    # plt.figure()
    plt.cla()
    # Loop over all paths
    for path in C_path:
        # Unzip the path into X and Y coordinate lists
        X, Y = zip(*path)
        # Plot this path
        plt.plot(X, Y)

    for x_obs, y_obs in obstacles:
        # plt.plot(x_obs, y_obs, "or", label="trajectory")
        radius =0.1
        circle = plt.Circle((x_obs, y_obs), radius, color='r', fill=True)
        plt.gcf().gca().add_artist(circle)

    # Loop over all trajectories
    for track in C_track:
        # Unzip the trajectory into X, Y, and theta lists
        for point in track:
            x,y = point
            plt.plot(x, y, ".b", label="trajectory")

        # Plot this trajectory
        # plt.plot(X1, Y1)

    for track in safe_trajectories:
        # Unzip the trajectory into X, Y, and theta lists
        for point in track:
            xs, ys = point
            plt.plot(xs, ys, ".y")

    for point in best_trajectory:
        xb, yb = point
        plt.plot(xb, yb, "go")

        # print(track)
        plt.axis("equal")
        plt.grid(True)

    plt.show()
