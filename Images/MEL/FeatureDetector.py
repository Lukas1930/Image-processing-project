import cv2
import numpy as np

def load_image_greyscale(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def load_image_colour(image_path):
    image = cv2.imread(image_path)
    return image

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

    # Create a dictionary to store the results
    color_stats = {
        "mean_blue": mean[0][0],
        "mean_green": mean[1][0],
        "mean_red": mean[2][0],
        "std_dev_blue": std_dev[0][0],
        "std_dev_green": std_dev[1][0],
        "std_dev_red": std_dev[2][0],
    }

    return color_stats

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

if __name__ == "__main__":
    image_path = "Masked\PAT_109_868_113_masked.png"
    grayscale_image = load_image_greyscale(image_path)
    image = load_image_colour(image_path)
    contour = find_contours(grayscale_image)
    symmetry_score = calculate_symmetry(grayscale_image, contour)
    print("Symmetry score:", symmetry_score)

    color_stats = color_contrast_and_variety(image, contour)
    print("Colour stats:", color_stats)

    compactness_value = compactness(contour)
    elongation_value = elongation(contour)
    roundness_value = roundness(contour)

    print("Compactness:", compactness_value)
    print("Elongation:", elongation_value)
    print("Roundness:", roundness_value)

    original_path = image_path.split('\\')[1].replace("_masked", "")
    original_greyscale = cv2.imread(original_path)
    original_greyscale = cv2.cvtColor(original_greyscale, cv2.COLOR_BGR2GRAY)

    border_sharpness_value = border_sharpness(original_greyscale, contour)
    print("Border sharpness:", border_sharpness_value)    