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

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Find an optimal threshold using Otsu's method and add a constant to raise tolerance
    threshold = filters.threshold_otsu(blurred_image) - 35

    # Segment the image using the threshold
    binary_image = blurred_image > threshold

    # Perform morphological closing operation to fill small gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_image = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Label the segmented regions
    labels = measure.label(closed_image)

    # Find the region with the largest area (assuming it's the skin lesion)
    largest_region = None
    max_area = 0
    for region in measure.regionprops(labels):
        if region.area > max_area:
            max_area = region.area
            largest_region = region

    # Create a mask for the largest region
    mask = np.zeros_like(closed_image)
    mask[labels == largest_region.label] = 1

    # Invert the mask
    inverted_mask = 1 - mask

    # Apply the inverted mask to the original image
    segmented_image = image.copy()
    segmented_image[inverted_mask == 0] = (0, 0, 0)

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