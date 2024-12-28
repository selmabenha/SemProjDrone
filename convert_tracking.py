from pathlib import Path
import torch
import numpy as np
import cv2
import gc
import os
import re
import logging
# from disp_utils import *
from utils import *

from PIL import Image
import argparse

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

logging.info("run script - transform")

# Paths
tracking_folder = "/home/finette/VideoStitching/transform_small"
frames_file = "/home/finette/VideoStitching/selma/output/textfiles/transform_recording/full_frames_list.txt"
transforms_file = "/home/finette/VideoStitching/selma/output/textfiles/transform_recording/all_transform_matrices.txt"


def process_iterations_with_matrices(full_frames_list, base_folder, transformation_matrices):
    for iteration, (frame_ranges, matrices) in enumerate(zip(full_frames_list, transformation_matrices), 1):
        print(f"Processing Iteration {iteration}...")
        
        # # Ensure the number of frame ranges matches the number of matrices
        # if len(frame_ranges) != 2 * len(matrices):
        #     raise ValueError(f"Mismatch between frame ranges and transformation matrices for iteration {iteration}.")

        # Process each pair of A and B ranges with corresponding [H, T, R]
        for idx, ((start_A, end_A), (start_B, end_B)) in enumerate(zip(frame_ranges[::2], frame_ranges[1::2])):
            H, T, R = matrices[idx]
            print(f"  Applying [H, T, R] set {idx + 1} for ranges A ({start_A}-{end_A}) and B ({start_B}-{end_B})")

            # Process A points
            for frame in range(start_A, end_A + 1):
                frame_path = f"{base_folder}/track_fr_{frame:04d}.txt"
                points = load_track_points(frame_path)

                # Transform each point as A
                modified_points = [
                    (np.array([transform_point_A(pt, H, T, R) for pt in point_set]), metadata)
                    for point_set, metadata in points
                ]
                save_track_points(frame_path, modified_points)

            # Process B points
            for frame in range(start_B, end_B + 1):
                frame_path = f"{base_folder}/track_fr_{frame:04d}.txt"
                points = load_track_points(frame_path)

                # Transform each point as B
                modified_points = [
                    (np.array([transform_point_B(pt, H, T, R) for pt in point_set]), metadata)
                    for point_set, metadata in points
                ]
                save_track_points(frame_path, modified_points)

        # Stop processing after the last meaningful iteration
        if iteration == len(full_frames_list) - 1:
            print(f"  Skipping iteration {iteration + 1} as it has no valid pairs.")
            break

frame_ranges, transformation_matrices = read_images_frames(frames_file, transforms_file)
process_iterations_with_matrices(frame_ranges, tracking_folder, transformation_matrices)