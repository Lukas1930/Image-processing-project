import os
import csv
import re

# Create a new CSV file
with open("output.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["Name", "Symmetry", "MeanBlue", "MeanGreen", "MeanRed",
                         "SDBlue", "SDGreen", "SDRed", "Compactness", "Elongation",
                         "Roundness", "Sharpness", "Label"])

    # Iterate over every folder in the root folder
    for folder_name in os.listdir("."):
        if os.path.isdir(folder_name):
            masked_folder = os.path.join(folder_name, "Masked")

            # Iterate over the text files in the Masked folder
            for file_name in os.listdir(masked_folder):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(masked_folder, file_name)

                    try:
                        with open(file_path, "r") as file:
                            lines = file.readlines()

                            # Ensure there are at least 6 lines
                            if len(lines) < 6:
                                print(f"Skipping {file_path} because it has less than 6 lines")
                                continue

                            # Extract required values from the text file
                            symmetry = float(lines[0].split()[0].strip(","))
                            color_stats = list(map(float, re.findall(r'[-+]?\d*\.\d+|\d+', lines[1])))

                            # Ensure there are 6 color stats
                            if len(color_stats) < 6:
                                print(f"Skipping {file_path} because it doesn't have 6 color stats")
                                continue

                            mean_blue, mean_green, mean_red, std_dev_blue, std_dev_green, std_dev_red = color_stats
                            compactness = float(lines[2].split()[0].strip(","))
                            elongation = float(lines[3].split()[0].strip(","))
                            roundness = float(lines[4].split()[0].strip(","))
                            sharpness = float(lines[5].split()[0].strip(","))

                            # Write values to the CSV file
                            csv_writer.writerow([file_name, symmetry,
                                                 mean_blue, mean_green, mean_red,
                                                 std_dev_blue, std_dev_green, std_dev_red,
                                                 compactness, elongation,
                                                 roundness, sharpness,
                                                 folder_name])
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")