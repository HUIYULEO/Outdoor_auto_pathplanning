"""Common utility functions"""
import numpy as np
import cv2 as cv
from pathlib import Path  

homo_path = Path(__file__).parent / 'opt_homoMatrix.npy'

def normalize_image(image, mean=None, std=None):
    """Normalize image with mean and std"""
    from . import config
    mean = mean or config.MEAN
    std = std or config.STD
    
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image


def resize_image(image, size):
    """Resize image to specified size"""
    return cv.resize(image, size, interpolation=cv.INTER_AREA)


def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    """Overlay binary mask on image"""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def angle_difference(angle1, angle2):
    """Calculate minimal angle difference between two angles (in degrees)"""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def point_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calcDist(cnt, dist):
    for i in range(0, len(cnt[0])):
        mind = 1e7
        for j in range(0, len(cnt[1])):
            temp = np.linalg.norm(cnt[0][i] - cnt[1][j])
            if temp < mind:
                mind = temp
                dist[0][i] = j

    for i in range(0, len(cnt[1])):
        mind = 1e7
        for j in range(0, len(cnt[0])):
            temp = np.linalg.norm(cnt[1][i] - cnt[0][j])
            if temp < mind:
                mind = temp
                dist[1][i] = j

    return dist


def merge(cnt, pairs):
    tempt = []
    tempb = []

    ###Open the bottom contour###
    x_r = 0
    x_l = 128
    idx_r = 0
    idx_l = 0

    for pair in pairs:
        if cnt[0][pair[0]][0, 0] > x_r:
            x_r = cnt[0][pair[0]][0, 0]
            idx_r = pair[0]
        if cnt[0][pair[0]][0, 0] < x_l:
            x_l = cnt[0][pair[0]][0, 0]
            idx_l = pair[0]

        if idx_r > idx_l:
            tempb = cnt[0][idx_l:idx_r]

        if idx_r < idx_l:
            tempb = np.concatenate((cnt[0][idx_l:len(cnt[0])], cnt[0][0:idx_r + 1]), axis=0)

    ###Open the top contour###
    x_r = 0
    x_l = 128

    for pair in pairs:
        if cnt[1][pair[1]][0, 0] > x_r:
            x_r = cnt[1][pair[1]][0, 0]
            idx_r = pair[1]
        if cnt[1][pair[1]][0, 0] < x_l:
            x_l = cnt[1][pair[1]][0, 0]
            idx_l = pair[1]

    if idx_r > idx_l:
        tempt = np.concatenate((cnt[1][idx_r:len(cnt[1])], cnt[1][0:idx_l + 1]), axis=0)

    if idx_r < idx_l:
        tempt = cnt[1][idx_r:idx_l]

    ###Merge the contours###
    if (len(tempb) > 0) and (len(tempt)) > 0:
        return [np.concatenate((tempb, tempt), axis=0)]
    else:
        if cv.contourArea(cnt[1]) > cv.contourArea(cnt[0]):
            return cnt[1]
        else:
            return cnt[0]


def removeSmallAreas(mask, threshold):
    """Remove small contours from mask"""
    result = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > threshold:
            filtered_contours.append(contour)
    
    mask = cv.drawContours(mask.copy(), filtered_contours, -1, 255, -1)
    
    return mask, filtered_contours

# Compatibility wrapper for removeSmallAreas 
def removeSmallAreas_compat(mask, threshold):
    """Call existing removeSmallAreas, fallback to a local implementation if it fails"""
    try:
        return removeSmallAreas(mask, threshold)
    except Exception:
        # fallback: robust findContours handling for OpenCV 3/4
        res = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(res) == 3:
            _, contours, _ = res
        else:
            contours, _ = res
        filtered = []
        for c in contours:
            if cv.contourArea(c) > threshold:
                filtered.append(c)
        mask2 = cv.drawContours(mask.copy(), filtered, -1, 255, -1)
        return mask2, filtered


def mergeContours(mask, contours):
    """Merge contours if needed
    
    Args:
        mask: binary mask
        contours: list of contours
    
    Returns:
        mask: updated mask
        contours: processed contours
    """
    dist = [None, None]
    dist[0] = [None] * len(contours[0])
    dist[1] = [None] * len(contours[1])

    pairs = []

    dist = calcDist(contours, dist)

    for i in range(len(dist[0])):
        if dist[1][dist[0][i]] == i:
            pairs.append([dist[1][dist[0][i]], dist[0][i]])

    if len(pairs) > 1:
        print("merge the contours")
        merged_cnt = merge(contours, pairs)

    else:
        if cv.contourArea(contours[1]) > cv.contourArea(contours[0]):
            merged_cnt = [contours[1]]
        else:
            merged_cnt = [contours[0]]

    largest_cnt = max(merged_cnt, key=cv.contourArea)

    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    new_mask = cv.drawContours(new_mask, [largest_cnt], -1, 255, thickness=cv.FILLED)

    return new_mask, largest_cnt


def img2BEV(image):
    height, width = 128, 128

    if not homo_path.exists():
        raise FileNotFoundError(f"Homography matrix not found at {homo_path}")

    h = np.load(homo_path)
    frameWarped = image.copy()
    frameWarped = cv.warpPerspective(frameWarped, h, (width, height))

    return frameWarped


def findLowerCorners(contour):
    lower_left = tuple(contour[0][0])
    lower_right = tuple(contour[0][0])

    for point in contour:
        if point[0][1] > lower_left[1] or (point[0][1] == lower_left[1] and point[0][0] < lower_left[0]):
            lower_left = tuple(point[0])

        if point[0][1] > lower_right[1] or (point[0][1] == lower_right[1] and point[0][0] > lower_right[0]):
            lower_right = tuple(point[0])

    return lower_left, lower_right


def test_find_edges(mask):
    """Find left and right edges of road in BEV
    
    Args:
        mask: binary mask in BEV view
    
    Returns:
        left_edge: list of points on left edge
        right_edge: list of points on right edge
    """
    left_edge = []
    right_edge = []

    for y in range(0, mask.shape[0], 8):
        white_pixels = np.where(mask[y, :] > 200)[0]
        if white_pixels.size > 0:
            left_edge.append((white_pixels[0], y))
            right_edge.append((white_pixels[-1], y))

    result = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(result) == 3:
        _, contours, _ = result
    else:
        contours, _ = result

    return left_edge, right_edge


def findFurthestMidpoint(left_edge, right_edge, bottom_midpoint):
    max_distance = 0
    furthest_midpoint = None

    for left_point, right_point in zip(left_edge, right_edge):
        midpoint = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
        distance = np.linalg.norm(np.array(bottom_midpoint) - np.array(midpoint))

        if distance > max_distance and midpoint[0] > left_point[0] and midpoint[0] < right_point[0]:
            max_distance = distance
            furthest_midpoint = midpoint

    return furthest_midpoint


def goalSearch(left_edge, right_edge, bottom_midpoint):
    max_distance = 0
    furthest_midpoint = None

    left_edge = sorted(left_edge, key=lambda point: -point[1])
    right_edge = sorted(right_edge, key=lambda point: -point[1])

    for left_point, right_point in zip(left_edge, right_edge):
        midpoint = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
        distance = np.linalg.norm(np.array(bottom_midpoint) - np.array(midpoint))

        if distance > max_distance:
            max_distance = distance
            furthest_midpoint = midpoint

    return furthest_midpoint


def calculate_midpoints(left_edge, right_edge):
    left_edge = sorted(left_edge, key=lambda point: point[1])
    right_edge = sorted(right_edge, key=lambda point: point[1])

    midpoints = []
    num_points = min(len(left_edge), len(right_edge))

    for i in range(num_points):
        midpoint_x = (left_edge[i][0] + right_edge[i][0]) // 2
        midpoint_y = (left_edge[i][1] + right_edge[i][1]) // 2
        midpoints.append((midpoint_x, midpoint_y))

    return midpoints


def pointcal(point):
    return [(point[0] - 63) / 17, (128 - point[1]) / 17]


def coordinateTrans(start, left_edge, right_edge, obstacle=None):
    start = pointcal(start)
    start = (start[0] - 0.5, start[1])

    l_edge = []
    r_edge = []
    for point_l in left_edge:
        point_l = pointcal(point_l)
        l_edge.append(point_l)
    for point_r in right_edge:
        point_r = pointcal(point_r)
        r_edge.append(point_r)

    if obstacle != None:
        r_edge.append(pointcal(obstacle))

    l_edge = np.array(l_edge)
    r_edge = np.array(r_edge)

    low_left = np.argmin(l_edge[:, 1])
    lpoint = l_edge[low_left]
    low_right = np.argmin(r_edge[:, 1])
    rpoint = r_edge[low_right]

    new_start = [(lpoint[0] + rpoint[0]) / 2, (lpoint[1] + rpoint[1]) / 2]

    top_center = findFurthestMidpoint(l_edge, r_edge, new_start)
    if top_center == None:
        top_center = (start[0], start[1] + 1)

    return new_start, top_center, l_edge, r_edge
