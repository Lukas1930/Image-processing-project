import cv2
import numpy as np
from skimage import filters, color, measure
import sys

def segment_skin_lesion(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Find an optimal threshold using Otsu's method
    threshold = filters.threshold_otsu(blurred_image)

    # Segment the image using the threshold
    binary_image = blurred_image > threshold

    # Label the segmented regions
    labels = measure.label(binary_image)

    # Find the region with the largest area (assuming it's the skin lesion)
    largest_region = None
    max_area = 0
    for region in measure.regionprops(labels):
        if region.area > max_area:
            max_area = region.area
            largest_region = region

    # Create a mask for the largest region
    mask = np.zeros_like(binary_image)
    mask[labels == largest_region.label] = 1

    # Invert the mask
    inverted_mask = 1 - mask

    # Apply the inverted mask to the original image
    segmented_image = image.copy()
    segmented_image[inverted_mask == 0] = (0, 0, 0)

    return segmented_image

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skin_lesion_segmentation.py <image_path>")
    else:
        image_path = sys.argv[1]
        segmented_image = segment_skin_lesion(image_path)
        cv2.imshow("Segmented Skin Lesion", segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()