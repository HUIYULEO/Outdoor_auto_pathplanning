import numpy as np

def jps(start, end, mask):
    width = mask.shape[1]
    height = mask.shape[0]

    # Define the heuristic function (Euclidean distance)
    def heuristic(node, goal):
        return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    # Define the cost function (Manhattan distance)
    def cost(node, neighbor):
        return np.abs(node[0] - neighbor[0]) + np.abs(node[1] - neighbor[1])

    def is_valid_jump_point(node, direction):
        dx = direction[0]
        dy = direction[1]
        x = node[0]
        y = node[1]

        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        if mask[y, x] == 0:
            return False

        if dx != 0 and dy != 0:
            if dx == -1 and dy == -1:
                if (x - 1 >= 0 and y - 1 >= 0 and mask[y, x - 1] == 0 and mask[y - 1, x] == 0):
                    return True
            elif dx == -1 and dy == 1:
                if (x - 1 >= 0 and y + 1 < height and mask[y, x - 1] == 0 and mask[y + 1, x] == 0):
                    return True
            elif dx == 1 and dy == -1:
                if (x + 1 < width and y - 1 >= 0 and mask[y, x + 1] == 0 and mask[y - 1, x] == 0):
                    return True
            elif dx == 1 and dy == 1:
                if (x + 1 < width and y + 1 < height and mask[y, x + 1] == 0 and mask[y + 1, x] == 0):
                    return True

        return False

    def get_jps_neighbors(current, goal):
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]

        neighbors = []

        # Check if the goal is reachable in the horizontal or vertical direction
        if dx == 0:
            x_direction = 0
        else:
            x_direction = dx // abs(dx)

        if dy == 0:
            y_direction = 0
        else:
            y_direction = dy // abs(dy)

        # Check for forced neighbors
        if dx != 0 or dy != 0:
            if dx == 0:
                neighbors.append((current[0], current[1] + y_direction))
                if not is_valid_jump_point((current[0], current[1] + y_direction), (0, y_direction)):
                    return neighbors
            elif dy == 0:
                neighbors.append((current[0] + x_direction, current[1]))
                if not is_valid_jump_point((current[0] + x_direction, current[1]), (x_direction, 0)):
                    return neighbors
            else:
                neighbors.append((current[0] + x_direction, current[1] + y_direction))

        # Check for diagonal neighbors
        if dx != 0 and dy != 0:
            neighbors.append((current[0] + x_direction, current[1]))
            neighbors.append((current[0], current[1] + y_direction))
            if not is_valid_jump_point((current[0] + x_direction, current[1]), (x_direction, 0)) or \
                    not is_valid_jump_point((current[0], current[1] + y_direction), (0, y_direction)):
                return neighbors

        # Recursive jump point search
        for neighbor in neighbors:
            if neighbor[0] == goal[0] and neighbor[1] == goal[1]:
                return neighbors

            dx = neighbor[0] - current[0]
            dy = neighbor[1] - current[1]
            forced_neighbors = get_forced_neighbors(neighbor, (dx, dy))
            neighbors.extend(forced_neighbors)

        return neighbors

    def get_forced_neighbors(current, direction):
        dx = direction[0]
        dy = direction[1]
        x = current[0]
        y = current[1]

        neighbors = []

        # Check if the node is within the image boundaries
        if x < 0 or x >= width or y < 0 or y >= height:
            return neighbors

        # Check if the node is reachable (white area in the mask)
        if mask[y, x] == 0:
            return neighbors

        # Check if the node has a forced neighbor
        if dx != 0 and dy != 0:
            if dx == -1 and dy == -1:
                if (x - 1 >= 0 and y - 1 >= 0 and mask[y, x - 1] == 0 and mask[y - 1, x] == 0):
                    neighbors.append((x - 1, y - 1))
            elif dx == -1 and dy == 1:
                if (x - 1 >= 0 and y + 1 < height and mask[y, x - 1] == 0 and mask[y + 1, x] == 0):
                    neighbors.append((x - 1, y + 1))
            elif dx == 1 and dy == -1:
                if (x + 1 < width and y - 1 >= 0 and mask[y, x + 1] == 0 and mask[y - 1, x] == 0):
                    neighbors.append((x + 1, y - 1))
            elif dx == 1 and dy == 1:
                if (x + 1 < width and y + 1 < height and mask[y, x + 1] == 0 and mask[y + 1, x] == 0):
                    neighbors.append((x + 1, y + 1))

        return neighbors

    # Initialize the open and closed sets
    open_set = set()
    closed_set = set()

    # Create a dictionary to store the parent of each node
    parent = {}

    # Create a dictionary to store the cost from the start node to each node
    g_score = {start: 0}

    # Create a dictionary to store the total cost from the start node to each node (g_score + heuristic)
    f_score = {start: heuristic(start, end)}

    # Add the start node to the open set
    open_set.add(start)

    # JPS algorithm
    while open_set:
        # Find the node with the minimum heuristic value
        current = min(open_set, key=lambda node: heuristic(node, end))

        # Check if the current node is the goal
        if current == end:
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

        # Get the jump point successors of the current node
        neighbors = get_jps_neighbors(current, end)

        for neighbor in neighbors:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + cost(current, neighbor)
            if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    # No path found
    return None
