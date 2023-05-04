import os
import shutil
import pandas as pd

# Read metadata.csv
metadata = pd.read_csv('metadata.csv')

# Set the path to your images folder
images_folder = os.getcwd()

# Iterate through the images in the folder
for img_file in os.listdir(images_folder):
    # Check if the file is an image with a .png extension
    if img_file.endswith('.png'):
        # Find the corresponding row in the metadata DataFrame
        name = img_file.replace("_masked", "")
        img_metadata = metadata.loc[metadata['img_id'] == name]

        # If a matching row is found
        if not img_metadata.empty:
            # Get the diagnostic value for this image
            diagnostic = img_metadata.iloc[0]['diagnostic']

            # Create the subfolder for this diagnostic value if it doesn't exist
            diagnostic_folder = os.path.join(images_folder, diagnostic)
            if not os.path.exists(diagnostic_folder):
                os.makedirs(diagnostic_folder)

            # Move the image to the corresponding subfolder
            src = os.path.join(images_folder, img_file)
            dst = os.path.join(diagnostic_folder, img_file)
            shutil.move(src, dst)
        else:
            print(f"No metadata found for image: {img_file}")