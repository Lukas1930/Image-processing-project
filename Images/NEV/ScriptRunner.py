import os
import sys
import subprocess
from pathlib import Path

# Replace this with the name of your existing script
script_name = 'Segmentator.py'

# Supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico')

def main():
    # Get the current working directory
    cwd = Path(os.getcwd())

    # Iterate over all the files in the directory
    for file in cwd.iterdir():
        # Check if the file has a supported image extension
        if file.is_file() and file.suffix.lower() in image_extensions:
            # Run your existing script with the image path as an argument
            subprocess.run([sys.executable, script_name, str(file)])

if __name__ == '__main__':
    main()