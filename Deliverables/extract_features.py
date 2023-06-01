import cv2
import numpy as np
from skimage import measure
import sys
import os

def extract_features(image):
    segmented_image = segment_skin_lesion(image)

    values = get_feature_values(image, segmented_image)

    return np.array(values, dtype=np.double)

def get_feature_values(original_image, segmented_image):
    grayscale_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    contour = find_contours(grayscale_segmented_image)

    #symmetry_score = calculate_symmetry(grayscale_segmented_image, contour) //Axed after chi tests
    mean_blue, mean_green, mean_red, std_dev_blue, std_dev_green, std_dev_red = color_contrast_and_variety(segmented_image, contour)
    compactness_value = compactness(contour)
    #elongation_value = elongation(contour) //Axed after chi tests
    roundness_value = roundness(contour)

    grayscale_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #border_sharpness_value = border_sharpness(grayscale_original_image, contour) //Axed after chi tests

    return mean_green, mean_red, std_dev_green, std_dev_red, compactness_value, roundness_value

def find_contours(grayscale_image):
    _, thresholded = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)

def calculate_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def split_lesion(grayscale_image, center):
    height, width = grayscale_image.shape
    left_half = grayscale_image[0:height, 0:center[0]]
    right_half = grayscale_image[0:height, center[0]:width]
    top_half = grayscale_image[0:center[1], 0:width]
    bottom_half = grayscale_image[center[1]:height, 0:width]
    return left_half, right_half, top_half, bottom_half

def calculate_symmetry(grayscale_image, contour):
    center = calculate_center(contour)
    left_half, right_half, top_half, bottom_half = split_lesion(grayscale_image, center)

    left_contour = find_contours(left_half)
    right_contour = find_contours(right_half)
    top_contour = find_contours(top_half)
    bottom_contour = find_contours(bottom_half)

    left_area = cv2.contourArea(left_contour)
    right_area = cv2.contourArea(right_contour)
    top_area = cv2.contourArea(top_contour)
    bottom_area = cv2.contourArea(bottom_contour)

    horizontal_symmetry = abs(left_area - right_area) / max(left_area, right_area)
    vertical_symmetry = abs(top_area - bottom_area) / max(top_area, bottom_area)

    symmetry_score = (horizontal_symmetry + vertical_symmetry) / 2
    return symmetry_score

def color_contrast_and_variety(image, contour):
    # Create a mask for the lesion
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the mean and standard deviation of the pixel intensities in each channel
    mean, std_dev = cv2.meanStdDev(masked_image, mask=mask)

    mean_blue = mean[0][0]
    mean_green = mean[1][0]
    mean_red = mean[2][0]
    std_dev_blue = std_dev[0][0]
    std_dev_green = std_dev[1][0]
    std_dev_red = std_dev[2][0]

    return mean_blue, mean_green, mean_red, std_dev_blue, std_dev_green, std_dev_red

def compactness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    compactness_value = 4 * np.pi * area / (perimeter ** 2)
    return compactness_value

def elongation(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(max(w, h)) / min(w, h)
    return aspect_ratio

def roundness(contour):
    area = cv2.contourArea(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius ** 2
    roundness_value = area / circle_area
    return roundness_value

def border_sharpness(grayscale_image, contour, num_samples=100):
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    contour_length = len(contour)
    step = contour_length // num_samples

    gradients = []

    for i in range(0, contour_length, step):
        point_on_contour = contour[i][0]
        x, y = point_on_contour[0], point_on_contour[1]
        gradients.append(gradient_magnitude[y, x])

    avg_gradient = np.mean(gradients)
    return avg_gradient

def segment_skin_lesion(image):
    # Apply a bilateral filter to reduce noise while preserving edges
    filtered_image = cv2.bilateralFilter(image, 15, 20, 20)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)

    # Calculate the average skin color by sampling the perimeters of the blurred image
    border_size = 20
    top_border = blurred_image[:border_size, :]
    bottom_border = blurred_image[-border_size:, :]
    left_border = blurred_image[border_size:-border_size, :border_size]
    right_border = blurred_image[border_size:-border_size, -border_size:]

    borders = np.concatenate([top_border.flatten(), bottom_border.flatten(), left_border.flatten(), right_border.flatten()])
    borders = borders.reshape(-1, 3)
    avg_skin_color = np.mean(borders, axis=0)

    # Calculate the average lesion color by sampling the central region of the blurred image
    central_size = 30
    h, w, _ = blurred_image.shape
    central_region = blurred_image[h // 2 - central_size // 2:h // 2 + central_size // 2,
                                   w // 2 - central_size // 2:w // 2 + central_size // 2]
    avg_lesion_color = np.mean(central_region.reshape(-1, 3), axis=0)

    # Calculate the absolute difference between the blurred image and the average skin color and average lesion color
    skin_diff_image = np.abs(blurred_image.astype(np.float32) - avg_skin_color.astype(np.float32))
    skin_diff_image = np.sum(skin_diff_image, axis=2)
    lesion_diff_image = np.abs(blurred_image.astype(np.float32) - avg_lesion_color.astype(np.float32))
    lesion_diff_image = np.sum(lesion_diff_image, axis=2)

    # Create a binary mask based on both the skin and lesion difference images
    skin_threshold = 125
    lesion_threshold = 190
    binary_image = np.logical_and(skin_diff_image > skin_threshold, lesion_diff_image < lesion_threshold)

    # Perform morphological closing operation to fill small gaps and remove noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    closed_image = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel)

    # Perform morphological opening operation to remove small regions like hair
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, opening_kernel)

    # Label the segmented regions
    labels = measure.label(opened_image)

    # Find the region with the largest area (assuming it's the skin lesion)
    largest_region = None
    max_area = 0
    for region in measure.regionprops(labels):
        # Remove small regions or regions with high eccentricity
        if region.area < 100 or region.eccentricity > 0.99:
            labels[labels == region.label] = 0
        elif region.area > max_area:
            max_area = region.area
            largest_region = region
            
    # Create a mask for the largest region
    mask = np.zeros_like(closed_image)
    mask[labels == largest_region.label] = 1

    # Invert the mask (masks the lesion for debugging purposes)
    #inverted_mask = 1 - mask

    # Apply the inverted mask to the original image
    segmented_image = image.copy()
    #segmented_image[inverted_mask == 0] = (0, 0, 0)
    segmented_image[mask == 0] = (0, 0, 0)

    return segmented_image