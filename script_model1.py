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
    # Load image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    cropped_image = image[:, :image.shape[1] - 120]  # crop right side
    blurred = cv2.GaussianBlur(cropped_image, (5, 5), 0)  # gaussian blur
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)  # initial threshold

    # Find the largest contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create an image for drawing contours
    image_with_contours = np.copy(image)

    # Draw the largest contour on the image_with_contours
    cv2.drawContours(image_with_contours, [largest_contour], -1, (255, 255, 255), 1)

    # Draw convexity defects on the image with contours
    defects_image = np.copy(image_with_contours)

    # Calculate convexity defects
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, hull)

    # Sort defects based on depth
    defects = sorted(defects, key=lambda x: x[0, 3], reverse=True)

    # Choose the far points with the lowest and third lowest y-coordinates
    if len(defects) >= 4:
        far_points = [tuple(largest_contour[defects[i][0][2]][0]) for i in range(4)]
        far_points = sorted(far_points, key=lambda point: point[1])  # Sort by y-coordinate
        first_defect_far, third_defect_far = far_points[0], far_points[2]

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


        # Calculate the length of the side of the ROI square
        length = int(np.sqrt((third_defect_far[0] - first_defect_far[0])**2 + (third_defect_far[1] - first_defect_far[1])**2))

        # Calculate the coordinates of the square vertices
        square_vertices = [
            (x_perpendicular + 50, y_perpendicular),
            (x_perpendicular + 50, y_perpendicular - length),
            (x_perpendicular + 50 + length, y_perpendicular - length),
            (x_perpendicular + 50 + length, y_perpendicular)
        ]

        # Ensure that midpoint contains integers
        midpoint = (int(midpoint[0]), int(midpoint[1]))

        # Calculate the angle of rotation
        angle = np.arctan2(-dy, dx) * 180 / np.pi

        # Create a rotation matrix
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

        # Convert the lists to NumPy arrays
        translated_along_perpendicular = np.array(translated_along_perpendicular, dtype=np.float32)
        square_vertices = np.array(square_vertices, dtype=np.float32)

        # Ensure a consistent order of points for perspective transformation
        rectified_order = np.array([[0, 0], [length, 0], [length, length], [0, length]], dtype=np.float32)

        # Perform a perspective transformation to rectify the rotated square to a rectangle
        transform_matrix = cv2.getPerspectiveTransform(translated_along_perpendicular, rectified_order)
        rectified_image = cv2.warpPerspective(image, transform_matrix, (length, length))
        rectified_image_equalized = cv2.equalizeHist(rectified_image)


        g_kernel_size = 5
        g_sigma = 2.5
        g_theta = np.pi / 3
        g_lambda = 8.0
        g_gamma = 0.4
        g_psi = 0.0

        # Create the Gabor kernel
        gabor_kernel = cv2.getGaborKernel((g_kernel_size, g_kernel_size), g_sigma, g_theta, g_lambda, g_gamma, g_psi, ktype=cv2.CV_32F)
        filtered_veins = cv2.filter2D(rectified_image_equalized, cv2.CV_32F, gabor_kernel)

        # Normalize the filtered image
        filtered_veins = cv2.normalize(filtered_veins, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply thresholding and CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        clahe_veins = clahe.apply(filtered_veins)
        clahe_blurred = cv2.GaussianBlur(clahe_veins, (5, 5), 0) # gaussian blur
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        clahe_veins = clahe.apply(clahe_blurred)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_veins = clahe.apply(clahe_veins)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        clahe_veins = clahe.apply(clahe_veins)
        _, binary_veins = cv2.threshold(clahe_veins, 110, 255, cv2.THRESH_BINARY)

        resized_image = cv2.resize(binary_veins, (128,128))

        if not ret:
            # Write the marked image to a new file
            marked_file_path = file_path.replace(original_image_directory, destination_image_directory)
            if os.path.exists(marked_file_path):
                os.remove(marked_file_path)  
            cv2.imwrite(marked_file_path, resized_image)
        else:
            return resized_image


def train_mode(base_directory, original_image_directory, destination_image_directory):
    selected_hand = "l"
    selected_spectrum = "940"
    for selected_person in range(1, 101):
        formatted_number = f"{selected_person:03d}"
        pattern = f"{formatted_number}_{selected_hand}_{selected_spectrum}_*.jpg"
        print(f"Processing image {pattern}.")

        matching_files = glob.glob(os.path.join(base_directory, original_image_directory, pattern))
        if matching_files:
            os.makedirs(os.path.join(base_directory, destination_image_directory), exist_ok=True)

            for idx, file_path in enumerate(matching_files):
                process_image(file_path, False, original_image_directory, destination_image_directory)
        else:
            print(f"No matching files found for person {formatted_number}.")

def save_trained_images(trained_images, filename="trained_images_model1.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(trained_images, file)

def load_trained_images(filename="trained_images_model1.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return {}

def match_mode(trained_data_path, image_to_match_path):
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