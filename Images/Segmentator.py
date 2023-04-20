import cv2
import numpy as np
from skimage import filters, measure
import sys
import os
from skimage.color import rgb2gray
from skimage.segmentation import active_contour
from skimage.filters import gaussian

def segment_skin_lesion(image_path):
    # Load the image
    image = cv2.imread(image_path)

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

    # Invert the mask
    #inverted_mask = 1 - mask

    # Apply the inverted mask to the original image
    segmented_image = image.copy()
    #segmented_image[inverted_mask == 0] = (0, 0, 0)
    segmented_image[mask == 0] = (0, 0, 0)

    return segmented_image

def save_masked_image(image_path, segmented_image):
    # Create the output directory if it doesn't exist
    output_dir = 'Masked'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the original file name without extension and add '_masked' to it
    file_name = os.path.splitext(os.path.basename(image_path))[0] + '_masked'

    # Get the extension of the original image
    extension = os.path.splitext(image_path)[1]

    # Combine the output directory, modified file name, and extension
    output_path = os.path.join(output_dir, file_name + extension)

    # Save the segmented image
    cv2.imwrite(output_path, segmented_image)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skin_lesion_segmentation.py <image_path>")
    else:
        image_path = sys.argv[1]
        segmented_image = segment_skin_lesion(image_path)

        # Save the segmented image
        save_masked_image(image_path, segmented_image)

        #cv2.imshow("Segmented Skin Lesion", segmented_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()