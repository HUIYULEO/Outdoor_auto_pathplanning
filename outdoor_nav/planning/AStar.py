import numpy as np
import cv2 as cv


# A* algorithm
def astar(start, end, mask):
    width = mask.shape[1]
    height = mask.shape[0]
    # Define the heuristic function (Euclidean distance)
    def heuristic(node, goal):
        return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    # Define the cost function (Manhattan distance)
    def cost(node, neighbor):
        return np.abs(node[0] - neighbor[0]) + np.abs(node[1] - neighbor[1])

    # Initialize the open and closed sets
    open_set = set()
    closed_set = set()

    # Create a dictionary to store the parent of each node
    parent = {}

    # Create a dictionary to store the cost from the start node to each node
    g_score = {start: 0}

    # Create a dictionary to store the total cost
    # from the start node to each node (g_score + heuristic)
    f_score = {start: heuristic(start, end)}

    # Add the start node to the open set
    open_set.add(start)

    # A* algorithm
    while open_set:
        # Find the node with the lowest f_score
        current = min(open_set, key=lambda node: f_score[node])

        # Check if the current node is the goal
        if current == end or cost(current, end) < 10:
            # Reconstruct the path
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        # Remove the current node from the open set and add it to the closed set
        open_set.remove(current)
        closed_set.add(current)

        step_length = 2

        # Explore the neighbors of the current node
        for neighbor in [(current[0] - step_length, current[1]), (current[0] + step_length, current[1]), (current[0], current[1] - step_length),
                         (current[0], current[1] + step_length), (current[0] - step_length, current[1] - step_length), (current[0] - step_length, current[1] + step_length),
                         (current[0] + step_length, current[1] - step_length), (current[0] + step_length, current[1] + step_length)]:
            # Check if the neighbor is valid (within the image boundaries and not in the closed set)
            if neighbor[0] >= 0 and neighbor[0] < width and neighbor[1] >= 0 and neighbor[1] < height and neighbor not in closed_set:

                # Check if the neighbor is reachable (white area in the mask)
                if (mask[neighbor[1], neighbor[0]] != 0):
                    tentative_g_score = g_score[current] + cost(current, neighbor)

                    # Check if the neighbor is not in the open set or the tentative g_score is lower than the current g_score
                    if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                        # Update the parent, g_score, and f_score of the neighbor
                        parent[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        d = heuristic(neighbor, end)
                        # w = 1
                        if d > 50:
                            w = 3
                        else:
                            w = 0.8
                        f_score[neighbor] = g_score[neighbor] + w * d

                        # Add the neighbor to the open set
                        open_set.add(neighbor)


    # No path found
    return None

