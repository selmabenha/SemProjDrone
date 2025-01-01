from pathlib import Path
import numpy as np
import cv2
import gc
import os
import logging
import re
from PIL import Image
import ast

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

def get_images_frames(image_paths):
    # Create a list of tuples (frame_number, image_path)
    image_with_frame_numbers = []
    images_cv = []
    frame_ranges = []
    # Extract frame numbers and file paths
    for path in image_paths:
        match = re.search(r'frame_(\d+)\.jpg', str(path))  # Match the frame number part
        if match:
            frame_number = int(match.group(1))  # Extract the frame number as an integer
            image_with_frame_numbers.append((frame_number, path))

    # Define the logic for frame ranges and add images to the list
    for i, (frame_number, path) in enumerate(image_with_frame_numbers):
        image = cv2.imread(str(path))

        # Resize the image
        height, width, _ = image.shape
        reduced_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

        # Determine the frame number range for the image
        if i == len(image_with_frame_numbers) - 1:
            frame_range = (frame_number, 7516)  # Last image, map it to the end CHANGE LATERRRRRRRRRRRRR
        else:
            frame_range = (frame_number, image_with_frame_numbers[i + 1][0] - 1)  # Normal case

        # Append the image and frame range to the respective lists
        images_cv.append(reduced_image)
        frame_ranges.append(frame_range)

    return images_cv, frame_ranges


def write_images_frames(full_frames_list, all_transform_matrices, output_dir="/home/finette/VideoStitching/selma/output/textfiles/transform_recording"):
    """
    Saves the full_frames_list and all_transform_matrices to text files.
    
    Args:
        full_frames_list (list of lists of tuples): List of all frame pairs per iteration.
        all_transform_matrices (list of lists): List of transform matrices per iteration.
        output_dir (str): Directory to save the output text files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write full_frames_list to a file
    frames_file_path = os.path.join(output_dir, "full_frames_list.txt")
    with open(frames_file_path, "w") as frames_file:
        for iteration, frame_pairs in enumerate(full_frames_list):
            frames_file.write(f"Iteration {iteration + 1}:\n")
            for frame_pair in frame_pairs:
                frames_file.write(f"  {frame_pair}\n")
            frames_file.write("\n")
    print(f"Full frames list saved to {frames_file_path}")
    
    # Write all_transform_matrices to a file
    matrices_file_path = os.path.join(output_dir, "all_transform_matrices.txt")
    with open(matrices_file_path, "w") as matrices_file:
        for iteration, transform_sets in enumerate(all_transform_matrices):
            matrices_file.write(f"Iteration {iteration + 1}:\n")
            for transform_set in transform_sets:
                # Convert the NumPy array to a plain list
                if isinstance(transform_set, np.ndarray):
                    matrices_file.write(f"  {transform_set.tolist()}\n")
                else:
                    matrices_file.write(f"  {transform_set}\n")
            matrices_file.write("\n")
    print(f"All transform matrices saved to {matrices_file_path}")

def read_images_frames(frames_file_path, matrices_file_path):
    """
    Reads full_frames_list and all_transform_matrices from text files.
    
    Args:
        frames_file_path (str): Path to the file containing the full frames list.
        matrices_file_path (str): Path to the file containing the transform matrices.
    
    Returns:
        tuple: A tuple containing:
            - full_frames_list (list of lists of tuples): The reconstructed frames list.
            - all_transform_matrices (list of lists): The reconstructed transform matrices.
    """
    full_frames_list = []
    all_transform_matrices = []
    # Read the full_frames_list from the file
    with open(frames_file_path, "r") as frames_file:
        current_iteration = []
        for line in frames_file:
            line = line.strip()
            if line.startswith("Iteration"):
                if current_iteration:
                    full_frames_list.append(current_iteration)
                    current_iteration = []
            elif line.startswith("("):  # This line contains a frame pair
                frame_pair = eval(line)  # Convert string to tuple
                current_iteration.append(frame_pair)
        if current_iteration:  # Append the last iteration
            full_frames_list.append(current_iteration)

    # Read the all_transform_matrices from the file
    with open(matrices_file_path, "r") as matrices_file:
        current_iteration = []
        for line in matrices_file:
            line = line.strip()
            if line.startswith("Iteration"):
                if current_iteration:
                    all_transform_matrices.append(current_iteration)
                    current_iteration = []
            elif line.startswith("["):  # This line contains a transform set
                # Use literal_eval to safely evaluate the list-like string
                try:
                    transform_set = ast.literal_eval(line)
                    # Convert the lists back to NumPy arrays
                    transform_set = [np.array(mat) for mat in transform_set]
                    current_iteration.append(transform_set)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue  # Skip the faulty line if any error occurs
        if current_iteration:  # Append the last iteration
            all_transform_matrices.append(current_iteration)

    return full_frames_list, all_transform_matrices

def load_track_points(file_path):
    """Load points and metadata from a tracking file."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 10:
                print(f"Invalid line in {file_path}: {line.strip()}")
                continue
            # Parse points and metadata
            try:
                points = np.array(list(map(float, parts[:8]))).reshape(4, 2)  # x1, y1, x2, y2, x3, y3, x4, y4
                metadata = parts[8:]  # name and id
                data.append((points, metadata))
            except ValueError as e:
                print(f"Error parsing line in {file_path}: {line.strip()} - {e}")
    return data

def save_track_points(file_path, data):
    """Save transformed points and metadata back to a file."""
    with open(file_path, 'w') as file:
        for points, metadata in data:
            # Flatten points array and format as strings
            points_flat = ", ".join(f"{coord:.6f}" for coord in points.flatten())
            metadata_str = ", ".join(metadata)
            file.write(f"{points_flat}, {metadata_str}\n")


def transform_first_point(A, H, translation_matrix, rotation_matrix):
    # Ensure A is a 2D point (x, y)
    A = np.array([A], dtype=np.float32)  # Shape (1, 2)
    print(f"A (2D point): {A.shape}")  # Debugging A shape

    # Convert to homogeneous coordinates by adding a third coordinate (1)
    A_homogeneous = np.hstack([A, np.ones((A.shape[0], 1), dtype=np.float32)])  # Shape (1, 3)
    print(f"A_homogeneous (homogeneous coordinates): {A_homogeneous.shape}")  # Debugging homogeneous shape

    # Ensure the transformation matrix H is 3x3
    H = np.array(H, dtype=np.float32)
    print(f"H (transformation matrix): {H.shape}")  # Debugging H shape

    # Apply perspective transformation (reshape to (1, 1, 3) for a single point)
    transformed_A = cv2.perspectiveTransform(A_homogeneous.reshape(1, 1, 3), H)  # Apply H
    print(f"transformed_A (after perspective transform): {transformed_A.shape}")  # Debugging transformed_A shape

    # Create full 3x3 translation matrix
    full_translation_matrix = np.eye(3, dtype=np.float32)
    full_translation_matrix[:2, 2] = translation_matrix[:, 2]

    # Apply translation to transformed_A
    transformed_A_homogeneous = np.dot(full_translation_matrix, np.append(transformed_A[0][0], 1))
    final_A = transformed_A_homogeneous[:2] / transformed_A_homogeneous[2]

    return final_A.astype(int)

def transform_second_point(B, H, translation_matrix, rotation_matrix):
# Transform point B (image1)
    # Step 1: Rotate point B
    rotated_B = np.dot(rotation_matrix, np.append(B, 1))[:2]  # Apply rotation matrix

    # Step 2: Translate rotated B
    B_homogeneous = np.append(rotated_B, 1)  # Convert to homogeneous coordinates
    transformed_B_homogeneous = np.dot(translation_matrix, B_homogeneous)  # Apply T
    final_B = transformed_B_homogeneous[:2]

    return final_B.astype(int)

def transform_first_points(points, H, T, R, S):
    # Convert points to numpy arrays if they are lists
    points = np.array(points, dtype=np.float32)
    H,T,R,S = np.array(H),np.array(T),np.array(R),np.array(S)

    # Scaling applied to pointsA
    points_scaled = points #/ S[0]
    # Homography transformation (H) applied to pointsA
    points_homogeneous = np.hstack([points_scaled, np.ones((points_scaled.shape[0], 1))])  # Convert points to homogeneous coordinates
    points_transformed = (H.dot(points_homogeneous.T)).T  # Apply homography matrix H to points
    points_transformed = points_transformed[:, :2] / points_transformed[:, 2:3]  # Normalize by the third coordinate (homogeneous division)
    # # Apply translation (T) to the transformed pointsA
    points_transformed[:, 0] += T[0, 2]
    points_transformed[:, 1] += T[1, 2]

    return points_transformed.astype(int)



def transform_second_points(points, H, T, R, S):
    """
    Transforms multiple points using rotation and translation matrices.

    Args:
        points (list or array): Array of 2D points to be transformed, shape (N, 2).
        H (ndarray): Homogeneous transformation matrix (3x3), currently unused but preserved for compatibility.
        translation_matrix (ndarray): Translation matrix (3x3).
        rotation_matrix (ndarray): Rotation matrix (3x3).

    Returns:
        ndarray: Transformed points, shape (N, 2).
    """
    # Convert points to numpy arrays if they are lists
    points = np.array(points, dtype=np.float32)
    H,T,R,S = np.array(H),np.array(T),np.array(R),np.array(S)

    # Scaling applied to pointsB
    points_scaled = points #/ S[1]
    # Apply rotation (R) to pointsB
    points_homogeneous = np.hstack([points_scaled, np.ones((points_scaled.shape[0], 1))])  # Convert points to homogeneous coordinates
    points_rotated = (R.dot(points_homogeneous.T)).T  # Apply rotation matrix R to points
    points_transformed = points_rotated #[:, :2] / pointsB_rotated[:, 2:3]  # Normalize by the third coordinate (homogeneous division)
    # Apply translation (T) to pointsB
    points_transformed[:, 0] += T[0, 2]
    points_transformed[:, 1] += T[1, 2]
    return points_transformed.astype(int)

def transform_points(pointsA, pointsB, H, T, R, S):
    """
    Transforms sets of points using a precomputed homography, translation matrix,
    and rotating the points of the second image using a provided rotation matrix.

    Args:
        points0 (list or ndarray): Points from the first image, shape (N, 2).
        points1 (list or ndarray): Points from the second image, shape (M, 2).
        H (ndarray): Precomputed homography matrix (3x3).
        translation_matrix (ndarray): Precomputed translation matrix (3x3).
        rotation_matrix (ndarray): Precomputed rotation matrix (2x3).

    Returns:
        transformed_points0 (ndarray): Transformed points from image0, shape (N, 2).
        transformed_points1 (ndarray): Transformed and rotated points from image1, shape (M, 2).
    """
    # Convert points to numpy arrays if they are lists
    pointsA = np.array(pointsA, dtype=np.float32)
    pointsB = np.array(pointsB, dtype=np.float32)
    H,T,R,S = np.array(H),np.array(T),np.array(R),np.array(S)

    # Scaling applied to pointsA
    pointsA_scaled = pointsA #/ S[0]
    # Homography transformation (H) applied to pointsA
    pointsA_homogeneous = np.hstack([pointsA_scaled, np.ones((pointsA_scaled.shape[0], 1))])  # Convert points to homogeneous coordinates
    pointsA_transformed = (H.dot(pointsA_homogeneous.T)).T  # Apply homography matrix H to points
    pointsA_transformed = pointsA_transformed[:, :2] / pointsA_transformed[:, 2:3]  # Normalize by the third coordinate (homogeneous division)
    # # Apply translation (T) to the transformed pointsA
    pointsA_transformed[:, 0] += T[0, 2]
    pointsA_transformed[:, 1] += T[1, 2]

    # Scaling applied to pointsB
    pointsB_scaled = pointsB #/ S[1]
    # Apply rotation (R) to pointsB
    pointsB_homogeneous = np.hstack([pointsB_scaled, np.ones((pointsB_scaled.shape[0], 1))])  # Convert points to homogeneous coordinates
    pointsB_rotated = (R.dot(pointsB_homogeneous.T)).T  # Apply rotation matrix R to points
    pointsB_transformed = pointsB_rotated #[:, :2] / pointsB_rotated[:, 2:3]  # Normalize by the third coordinate (homogeneous division)
    # Apply translation (T) to pointsB
    pointsB_transformed[:, 0] += T[0, 2]
    pointsB_transformed[:, 1] += T[1, 2]

    return pointsA_transformed, pointsB_transformed
