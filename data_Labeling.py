import cv2
import os
import numpy as np
import csv

# === CONFIGURATION ===

# Folder containing your dataset of images (change this to your own dataset path)
dataset_path = "/path/to/images_light"  # <- UPDATE this if the folder is in a different location

# Path to output CSV file where the labels will be saved
labels_file = os.path.join(dataset_path, "labels.csv")

# Thresholds and conditions used for labeling
SATURATION_THRESHOLD = 230  # Pixels with value >= 230 are considered saturated
SATURATED_PERCENTAGE_THRESHOLD = 3.0  # % of saturated pixels required to flag as "blinded"
BRIGHT_BLOB_THRESHOLD = 240  # Threshold for bright spot detection via contouring
BLOB_AREA_THRESHOLD = 2000  # Minimum blob area to be considered a significant light source
MEAN_BRIGHTNESS_THRESHOLD = 130  # Average image brightness threshold

# === SCRIPT EXECUTION ===

# List all PNG files in the dataset directory
# Change '.png' to '.jpg' or another extension if needed
frames = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])

# Create the CSV file and write the header
with open(labels_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label', 'percentage'])  # CSV columns: image name, label (0/1), saturation %

    for frame in frames:
        image_path = os.path.join(dataset_path, frame)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale

        height, width = img.shape
        total_pixels = height * width

        # Calculate mean brightness and saturated pixel percentage
        mean_brightness = np.mean(img)
        saturated_pixels = np.sum(img >= SATURATION_THRESHOLD)
        saturated_percent = (saturated_pixels / total_pixels) * 100

        # Detect bright blobs using contouring (simulates flares or hotspots)
        _, thresh = cv2.threshold(img, BRIGHT_BLOB_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_blob_found = any(cv2.contourArea(cnt) >= BLOB_AREA_THRESHOLD for cnt in contours)

        # Label assignment: 1 = Blinded, 0 = Not Blinded
        if (
            mean_brightness >= MEAN_BRIGHTNESS_THRESHOLD and 
            saturated_percent >= SATURATED_PERCENTAGE_THRESHOLD
        ) or large_blob_found:
            label = 1
        else:
            label = 0

        # Save row to CSV
        writer.writerow([frame, label, saturated_percent])

        # Optional debug log per image
        print(f"{frame}: mean={mean_brightness:.2f}, sat%={saturated_percent:.2f}, blob={large_blob_found} -> label={label}")

print(f"✅ Smart labeling complete! Labels saved to: {labels_file}")

