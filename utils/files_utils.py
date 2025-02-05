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

def get_images_frames(image_paths, last_frame_num):
    """
    Given a list of image paths, this function processes the images and assigns a range of indices
    to each frame. It returns the resized images and the dynamically calculated indices for each frame.
    
    Parameters:
        image_paths (list): List of paths to the images.

    Returns:
        images_cv (list): List of resized images.
        frames_info (list): List of dictionaries containing frame indices.
    """
    image_with_frame_numbers = []
    images_cv = []
    frames_info = []

    # Extract frame numbers and file paths
    for path in image_paths:
        match = re.search(r'frame_(\d+)\.jpg', str(path))  # Match the frame number part
        if match:
            frame_number = int(match.group(1))  # Extract the frame number as an integer
            image_with_frame_numbers.append((frame_number, path))

    # Sort the image list by frame number to ensure correct ordering
    image_with_frame_numbers.sort(key=lambda x: x[0])

    # Define frame ranges dynamically
    for i, (frame_number, path) in enumerate(image_with_frame_numbers):
        image = cv2.imread(str(path))

        if image is None:
            print(f"Warning: Unable to read image at {path}")
            continue  # Skip this image if it's not read properly

        # Resize the image
        height, width, _ = image.shape
        reduced_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

        # Define indices dynamically
        if i == len(image_with_frame_numbers) - 1:
            frame_range = {"indices": list(range(frame_number, last_frame_num))}  # Last frame range
        else:
            next_frame_number = image_with_frame_numbers[i + 1][0]
            frame_range = {"indices": list(range(frame_number, next_frame_number))}

        # Append the image and frame indices to the respective lists
        images_cv.append(reduced_image)
        frames_info.append(frame_range)

    return images_cv, frames_info

def log_stitching(stitching_log, operation_id, frame_a, frame_b, result_frame, transformation_matrices):
    """
    Log the stitching operation with transformation matrices and frame indices.

    :param operation_id: Identifier for the stitching operation
    :param frame_a: First frame (with indices)
    :param frame_b: Second frame (with indices)
    :param result_frame: Resulting frame (with indices)
    :param transformation_matrices: List of transformation matrices used in this operation
    """
    entry = {
        "operation_id": operation_id,
        "input_frames": [
            {"frame_indices": frame_a["indices"]},
            {"frame_indices": frame_b["indices"]}
        ],
        "result_frame": {
            "frame_indices": result_frame["indices"]
        },
        "transformation_matrices": transformation_matrices  # Store as strings for clarity
    }
    stitching_log.append(entry)
    return stitching_log

def write_stitching_log(stitching_log, output_dir="output/textfiles/stitching_log.txt"):
    """
    Saves the stitching log to a text file.
    
    Args:
        stitching_log (list): The stitching log to be saved.
        output_dir (str): Directory to save the output text file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write stitching_log to a file
    log_file_path = os.path.join(output_dir, "stitching_log.txt")
    with open(log_file_path, "w") as log_file:
        for entry in stitching_log:
            operation_id = entry["operation_id"]
            input_frames = entry["input_frames"]
            result_frame = entry["result_frame"]
            transformation_matrices = entry["transformation_matrices"]

            # Write the stitching log entry
            log_file.write(f"Operation ID: {operation_id}\n")
            log_file.write("Input Frames:\n")
            for frame in input_frames:
                log_file.write(f"  Frame Indices: {frame['frame_indices']}\n")
            log_file.write(f"Resulting Frame Indices: {result_frame['frame_indices']}\n")
            log_file.write(f"Transformation Matrices: {transformation_matrices}\n")
            log_file.write("\n")
    
    print(f"Stitching log saved to {log_file_path}")

import os

def read_stitching_log(input_dir="output/textfiles/stitching_log.txt"):
    """
    Reads the stitching log from a text file and returns the parsed log entries.
    
    Args:
        input_dir (str): Directory to read the stitching log text file from.
    
    Returns:
        list: A list of dictionaries representing the stitching log.
    """
    log_file_path = os.path.join(input_dir, "stitching_log.txt")
    
    if not os.path.exists(log_file_path):
        print(f"Stitching log file not found at {log_file_path}")
        return []

    stitching_log = []
    
    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()
        entry = {}
        input_frames = []
        for line in lines:
            line = line.strip()

            # Check for Operation ID
            if line.startswith("Operation ID:"):
                if entry:
                    # If entry already exists, append it to the stitching_log
                    entry["input_frames"] = input_frames
                    stitching_log.append(entry)
                    # Reset for the next entry
                    entry = {}
                    input_frames = []
                entry["operation_id"] = line.split(":")[1].strip()

            # Check for Input Frames
            elif line.startswith("Input Frames:"):
                pass  # Skip this header line as it's just for readability
            elif line.startswith("Frame Indices:"):
                indices = line.split(":")[1].strip()
                indices = eval(indices)  # Convert string to list
                input_frames.append({"frame_indices": indices})

            # Check for Resulting Frame
            elif line.startswith("Resulting Frame Indices:"):
                result_frame_indices = line.split(":")[1].strip()
                entry["result_frame"] = {"frame_indices": eval(result_frame_indices)}

            # Check for Transformation Matrices
            elif line.startswith("Transformation Matrices:"):
                transformation_matrices = line.split(":")[1].strip()
                # Convert transformation matrices from string representation back to a list
                entry["transformation_matrices"] = eval(transformation_matrices)
        
        # Append the last entry
        if entry:
            entry["input_frames"] = input_frames
            stitching_log.append(entry)

    print(f"Stitching log read from {log_file_path}")
    return stitching_log

def write_track_file(txt_frame_path, tracker_res):
    # Open the text file in append mode ("a")
    with open(txt_frame_path, "a") as f:
        # Iterate over the tracker_res to get each tracked object
        for track_id, boxes in tracker_res.tracked_bboxes.items():
            # Extract bounding box (x, y, w, h)
            x, y, w, h = (int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
            
            # Calculate the 4 corners of the bounding box
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h
            
            # Get class name for this track id
            class_name = tracker_res.tracked_class_names[track_id]
            
            # Write the data to the file
            f.write(f"{x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}, {class_name}, {track_id}\n")


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
