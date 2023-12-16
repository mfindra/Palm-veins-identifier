import argparse
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.metrics as metrics
import re
import pickle


def process_image(file_path, ret=False, original_image_directory="", destination_image_directory=""):
    # Load the original image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Work data preprocessing
    cropped_image = image[:, :image.shape[1] - 120]
    blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Contouring hand
    image_with_contours = np.copy(image)
    cv2.drawContours(image_with_contours, [largest_contour], -1, (255, 255, 255), 1)

    # Calculate convexity defects
    defects_image = np.copy(image_with_contours)
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, hull)

    # Sort defects based on depth
    defects = sorted(defects, key=lambda x: x[0, 3], reverse=True)

    # Choose the far points with the lowest and third lowest y-coordinates
    far_points = [tuple(largest_contour[defects[i][0][2]][0]) for i in range(4)]
    far_points = sorted(far_points, key=lambda point: point[1])  # Sort by y-coordinate
    first_defect_far, third_defect_far = far_points[0], far_points[2]

    # Draw a line between the first and third defects
    cv2.line(defects_image, first_defect_far, third_defect_far, [0, 0, 255], 2)

    # Calculate the midpoint of the line
    midpoint = ((first_defect_far[0] + third_defect_far[0]) // 2, (first_defect_far[1] + third_defect_far[1]) // 2)

    # Calculate the direction vector (dx, dy) of the line
    dx = third_defect_far[0] - first_defect_far[0]
    dy = third_defect_far[1] - first_defect_far[1]

    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    dx /= length
    dy /= length

    # Calculate the coordinates of the perpendicular line
    x_perpendicular = int(midpoint[0] + 50 * dy)
    y_perpendicular = int(midpoint[1] - 50 * dx)

    # Draw the perpendicular line
    cv2.line(defects_image, midpoint, (x_perpendicular, y_perpendicular), [255, 0, 0], 2)

    # Calculate the length of the side of the square
    length = int(np.sqrt((third_defect_far[0] - first_defect_far[0])**2 + (third_defect_far[1] - first_defect_far[1])**2))

    # Calculate the coordinates of the square vertices
    square_vertices = [
        (x_perpendicular + 50, y_perpendicular),
        (x_perpendicular + 50, y_perpendicular - length),
        (x_perpendicular + 50 + length, y_perpendicular - length),
        (x_perpendicular + 50 + length, y_perpendicular)
    ]

    # Calculate the angle of rotation
    angle = np.arctan2(-dy, dx) * 180 / np.pi

    # Create a rotation matrix
    midpoint = (int(midpoint[0]), int(midpoint[1]))
    rotation_matrix = cv2.getRotationMatrix2D(midpoint, angle, scale=1)

    # Apply the rotation to the square vertices
    rotated_square_vertices = cv2.transform(np.array([square_vertices], dtype=np.float32), rotation_matrix).squeeze().astype(np.int32)

    # Calculate the new starting point for the square
    start_point = (first_defect_far[0] - rotated_square_vertices[0][0], first_defect_far[1] - rotated_square_vertices[0][1])

    # Translate the rotated square to the new starting point
    translated_square_vertices = rotated_square_vertices + start_point

    # Calculate the direction vector (dx_perpendicular, dy_perpendicular) of the perpendicular line
    dx_perpendicular = x_perpendicular - midpoint[0]
    dy_perpendicular = y_perpendicular - midpoint[1]

    # Normalize the direction vector
    length_perpendicular = np.sqrt(dx_perpendicular**2 + dy_perpendicular**2)
    dx_perpendicular /= length_perpendicular
    dy_perpendicular /= length_perpendicular

    # Calculate the translation vector along the perpendicular line
    translation_vector = (int(50 * dx_perpendicular), int(50 * dy_perpendicular))

    # Translate the rotated and aligned square vertices
    translated_along_perpendicular = translated_square_vertices + translation_vector

    # Draw the rotated, aligned, and translated square
    cv2.polylines(defects_image, [translated_along_perpendicular], isClosed=True, color=[255, 0, 0], thickness=2)

    # Convert the lists to NumPy arrays
    translated_along_perpendicular = np.array(translated_along_perpendicular, dtype=np.float32)
    square_vertices = np.array(square_vertices, dtype=np.float32)

    # Ensure a consistent order of points for perspective transformation
    rectified_order = np.array([[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32)

    # Perspective transformation to rectify the rotated square to a rectangle
    transform_matrix = cv2.getPerspectiveTransform(translated_along_perpendicular, rectified_order)
    rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))

    rectified_image_equalized = cv2.equalizeHist(rectified_image)
    
    # Gabor filter
    g_kernel_size = 4
    g_lambda = 10.0
    g_psi = 0.0

    gabor_images = []
    roi = rectified_image_equalized
    
    for g_theta in range(8):
        g_theta = g_theta / 4. * np.pi
        for g_sigma in (1, 3):
            kernel = cv2.getGaborKernel((g_kernel_size,g_kernel_size), g_sigma, g_theta, g_lambda, g_psi)
            roi_blur = cv2.GaussianBlur(roi, (9, 9), 2)
            filtered_image = cv2.filter2D(roi_blur, -1, kernel)
            gabor_images.append(filtered_image)

    # Normalize the filtered image for visualization
    gabor_output = np.average(gabor_images, axis=0)
    filtered_veins = cv2.normalize(gabor_output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create and apply a CLAHE (Contrast Limited Adaptive Histogram Equalization) object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    clahe_veins = clahe.apply(filtered_veins)
    clahe_veins_equalized = cv2.equalizeHist(clahe_veins)

    # Apply adaptive thresholding
    binary_veins = cv2.adaptiveThreshold(clahe_veins_equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
    mask2_blur = cv2.GaussianBlur(rectified_image, (9, 9), 2)

    # Apply thresholding and CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
    clahe_veins = clahe.apply(mask2_blur)
    clahe_blurred = cv2.GaussianBlur(clahe_veins, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    clahe_veins = clahe.apply(clahe_blurred)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_veins = clahe.apply(clahe_veins)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    clahe_veins = clahe.apply(clahe_veins)

    mask2_inv = cv2.adaptiveThreshold(cv2.equalizeHist(clahe_veins), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 5)

    kernel_size = 5 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Mask out of thrasholded values in original image
    _, mask1_inv = cv2.threshold(rectified_image_equalized, 180, 255, cv2.THRESH_BINARY_INV)

    # Apply mask 1
    result = cv2.bitwise_and(mask1_inv, mask1_inv, mask=cv2.bitwise_not(binary_veins))
    result_inverted = cv2.bitwise_not(result)

    # Apply mask 2
    result2 = cv2.bitwise_and(mask2_inv, mask2_inv, mask=cv2.bitwise_not(result_inverted))
    result2_inverted = cv2.bitwise_not(result2)

    # Perform morphological opening
    binary_veins_opened = cv2.morphologyEx(result2_inverted, cv2.MORPH_OPEN, kernel)

    resized_image = cv2.resize(binary_veins_opened, (128,128))

    if not ret:
        # Write the marked image to a new file
        marked_file_path = file_path.replace(original_image_directory, destination_image_directory)
        if os.path.exists(marked_file_path):
            os.remove(marked_file_path)  
        cv2.imwrite(marked_file_path, resized_image)
    else:
        return resized_image


def train_mode(base_directory, original_image_directory, destination_image_directory):
    # Replace 'selected_person' and 'selected_hand' with your chosen person and hand

    # Define the spectrum to focus on
    selected_hand = "l"
    selected_spectrum = "940"

    # Loop through selected persons
    for selected_person in range(1, 101):  # Assuming persons are numbered from 1 to 100
        formatted_number = f"{selected_person:03d}"
        

        # Define the pattern to match the file names
        pattern = f"{formatted_number}_{selected_hand}_{selected_spectrum}_*.jpg"

        print(f"Processing image {pattern}.")
        # Use glob to get a list of file paths that match the pattern
        matching_files = glob.glob(os.path.join(base_directory, original_image_directory, pattern))

        # Check if there are matching files
        if matching_files:
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.join(base_directory, destination_image_directory), exist_ok=True)

            # Loop through the selected files and create an outline
            for idx, file_path in enumerate(matching_files):
                process_image(file_path, False, original_image_directory, destination_image_directory)

        else:
            # Display a message if no matching files are found for the current person
            print(f"No matching files found for person {formatted_number}.")

def save_trained_images(trained_images, filename="trained_images_model2.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(trained_images, file)

def load_trained_images(filename="trained_images_model2.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return {}

def match_mode(trained_data_path, image_to_match_path):
    # Load the image to match
    image_to_match = process_image(image_to_match_path, True)

    # Load or create trained data
    trained_images = load_trained_images()
    if not trained_images:
        trained_images = {}
        for file_path in glob.glob(trained_data_path + "*.jpg"):
            match = re.search(r'(\d+)_(\w)_(\d+)_(\d+).jpg', file_path)
            if match:
                person_id, hand, spectrum, number = match.groups()
                key = f"{person_id}_{hand}_{spectrum}"
                if key not in trained_images:
                    trained_images[key] = []
                trained_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                trained_images[key].append((file_path, trained_image))

        # Save the trained data
        save_trained_images(trained_images)

    # Calculate similarities
    similarities = []
    for key, images in trained_images.items():

        for path, trained_image in images:
            ssim = metrics.structural_similarity(trained_image, image_to_match)

            similarities.append((key, ssim))

    # Sort similarities and print the top 5
    top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    for rank, (key, similarity) in enumerate(top_similarities, start=1):
        print(f"Rank {rank}: Group {key} similarity: {similarity:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('--train', nargs=3, help='Run in training mode: base_dir, original_dir, destination_dir', metavar=('base_dir', 'original_dir', 'destination_dir'))
    parser.add_argument('--match', nargs=2, help='Run in matching mode: trained_data_path, image_to_match_path', metavar=('trained_data_path', 'image_to_match_path'))

    args = parser.parse_args()

    if args.train:
        train_mode(*args.train)

    elif args.match:
        match_mode(*args.match)

    else:
        print("Error: Specify either --train or --match mode.")

if __name__ == "__main__":
    main()