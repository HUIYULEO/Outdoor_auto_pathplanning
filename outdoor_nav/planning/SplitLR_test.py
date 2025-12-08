import os
import torch
import cv2 as cv
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.Unet import UNet
from src.Unetplus import NestedUNet
from torchvision import transforms
from planner.AStar import astar
from archive.test.Utility import *

# Path to the folder containing the images
FOLDER = 'ImageSamples'
DEBUG_PATH = 'Dangerpic'  #'SpecialImage'

# Path to the saved PyTorch model
MODEL_unetpp = 'Trainedmodel/unetpp_100N.pth'

# image_path = 'testimg/img.png' # 8560
image_path = 'ImageSamples/23320.png' # 8560

def findLowerCorners(contour):
    # Initialize the lower left and lower right corners
    lower_left = tuple(contour[0][0])
    lower_right = tuple(contour[0][0])

    for point in contour:
        # Update the lower left corner if needed
        if point[0][1] > lower_left[1] or (point[0][1] == lower_left[1] and point[0][0] < lower_left[0]):
            lower_left = tuple(point[0])

        # Update the lower right corner if needed
        if point[0][1] > lower_right[1] or (point[0][1] == lower_right[1] and point[0][0] > lower_right[0]):
            lower_right = tuple(point[0])

    # Sorting contour by y coordinate (largest to smallest) for further processing
    sorted_contour_y_desc = sorted(np.squeeze(contour), key=lambda point: -point[1])

    return sorted_contour_y_desc, lower_left, lower_right

def splitContourPoints(contour, lower_left, lower_right):
    left_edge, right_edge = [lower_left], [lower_right]

    for point in contour[1:]:
        dist_left = np.linalg.norm(np.array(lower_left) - np.array(point))
        dist_right = np.linalg.norm(np.array(lower_right) - np.array(point))

        if dist_left < dist_right :
            lower_left = point
            left_edge.append(lower_left)
        else:
            lower_right = point
            right_edge.append(lower_right)

    return left_edge, right_edge

def homography(image, mask, contour):
    height, width = 480, 640
    hCorr, wCorr = 572, 720
    warpMultiplier = 35
    corrMat = np.eye(3)
    corrMat[0, 2] = corrMat[0, 2] + wCorr
    corrMat[1, 2] = corrMat[1, 2] + hCorr
    ptTrue = np.array([[720, 1080.0], [0, 1080.0], [0, 0], [720, 0]])
    pt = [[559, 474], [73, 474], [212, 323], [388, 323]]

    h, _ = cv.findHomography(np.array(pt), ptTrue)
    h = np.matmul(corrMat, h)

    frameWarped = image.copy()
    maskWarped = mask.copy()

    frameWarped = cv.warpPerspective(frameWarped, h, (int(width * warpMultiplier/10), int(height * warpMultiplier/10)))
    frameWarped = cv.resize(frameWarped, (width, height))

    maskWarped = cv.warpPerspective(maskWarped, h, (int(width * warpMultiplier/10), int(height * warpMultiplier/10)))
    maskWarped = cv.resize(maskWarped, (width, height))

    # Create an empty list to hold the warped contours
    warped_contours = []

    # Loop over each contour
    for contour in contours:
        # Convert the contour points to homogeneous coordinates
        # Reshape the contour array to the required 3D format
        contour = contour.reshape(-1, 1, 2)

        # Apply the homography matrix to the contour points
        warped_contour = cv.perspectiveTransform(contour.astype(np.float32), h)

        # Convert back to Cartesian coordinates
        warped_contour = warped_contour[:, 0, :].astype(int)

        # Add the warped contour to the list
        warped_contours.append(warped_contour)

    # cv.line(frame, (64, 0), (64, 128), (0, 0, 255), 1)
    cv.line(frameWarped, (320, 0), (320, 480), (0, 0, 255), 1)

    return frameWarped, maskWarped, warped_contour

if __name__ == '__main__':
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = UNet().to(device)
    model = NestedUNet().to(device)
    model.load_state_dict(torch.load('Trainedmodel/unetpp_100N.pth', map_location=torch.device('cpu')))
    # torch.no_grad()
    model.eval()

    test_data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    org_img = Image.open(image_path)
    img = test_data_transform(org_img).unsqueeze(0)

    with torch.no_grad():
        img_ = img.to(device)
        pred = model(img_)

    pred = F.sigmoid(pred)
    pred.detach().numpy()

    # Convert the PIL image to a NumPy array
    output_image = np.array(pred.squeeze(0).squeeze(0) * 255, dtype=np.uint8)
    output_image = Image.fromarray(output_image)
    output_image = output_image.resize((640, 480))

    # Convert the PIL image to a NumPy array with three channels
    output_image = np.array(output_image.convert('RGB'))

    # Display the combined image
    cv.namedWindow('Original Image', cv.WINDOW_AUTOSIZE)
    cv.resizeWindow('Original Image', 640, 480)
    cv.imshow("Original Image", cv.imread(image_path))

    threshold = 0.5  # Adjust this threshold as needed
    mask = (pred > threshold).float()  # Binarize the prediction
    mask = mask.squeeze(0).squeeze(0).numpy()
    mask = cv.resize(mask, (640, 480))
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask

    overlay_color = (150, 150, 150)  # White color

    # Apply the mask as an overlay on the original image
    result = cv.imread(image_path)
    result = cv.resize(result, (640, 480))
    result[np.where(mask)] = overlay_color

    # result, mask = homography(result, mask)

    # Find contours of the white area
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = removeSmallContours(contours, 100)
    # If any contours remain after noise removal
    if len(contours) > 0:

        # Generate approximate contour to reduce overall amount of data
        # for i in range(len(contours)):
        #     contours[i] = cv.approxPolyDP(contours[i], 2, True)

        # If exactly two contours remain in the image, try to merge them
        if len(contours) == 2:
            contours = mergeContours(contours)
            mask = cv.drawContours(mask, contours, 0, (255, 255, 255), cv.FILLED)

    result, mask, contours = homography(result, mask, contours)

    largest_contour = max(contours, key=cv.contourArea)

    # Find the lower left and lower right corners of the contour
    sorted_contour, lower_left, lower_right = findLowerCorners(largest_contour)

    cv.circle(result, lower_left, 2, (255, 0, 0), -1)
    cv.circle(result, lower_right, 2, (0, 255, 0), -1)

    # Convert the contour to a numpy array
    # LRcontour = np.squeeze(largest_contour)
    # Split the contour points into left and right edge point sets
    # left_edge, right_edge = splitContourPoints(LRcontour, lower_left, lower_right)
    left_edge, right_edge = splitContourPoints(sorted_contour, lower_left, lower_right)

    for left_point in left_edge:
        cv.circle(result, left_point, 2, (255, 255, 0), -1)

    for ritht_point in right_edge:
        cv.circle(result, ritht_point, 2, (255, 0, 255), -1)

    # Display the combined image
    cv.namedWindow('Original Image vs Segmentation', cv.WINDOW_AUTOSIZE)
    cv.imshow("Original Image vs Segmentation", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

