from pathlib import Path
import torch
import numpy as np
import cv2
import gc
import os
import re
import logging
from utils.files_utils import *

from PIL import Image
import argparse
import math

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

logging.info("run script - transform")

index_ref = 0

# import pandas as pd

def frame_pix_displacement(index, tracking_files, reference_frames):
    global index_ref
    if index == 0:
        return [(0.0, 0.0)]
    elif index == reference_frames[index_ref+1] and index != 1:
        index_ref += 1
        return [(0.0, 0.0)]
    else:
        # Load the points for the current and reference frames
        points_ref = load_track_points(Path(tracking_files[reference_frames[index_ref]]))
        points = load_track_points(Path(tracking_files[index]))
        if points_ref is None and points is None: raise ValueError("Issue with loading track points both")
        elif points is None: raise ValueError("Issue with loading track points")
        elif points_ref is None: raise ValueError("Issue with loading track points_ref")
        # Extract only the y-coordinates for points with matching class_id
        xy_displacements = []  # List to store the displacements
        for point in points:
            coords, metadata = point
            class_id = metadata[1].strip()  # Extract class_id from current point
            
            # Find matching class_id_ref in points_ref
            for point_ref in points_ref:
                coords_ref, metadata_ref = point_ref
                class_id_ref = metadata_ref[1].strip()  # Extract class_id_ref from reference point
                
                if class_id == class_id_ref:  # Check if class_id matches class_id_ref
                    y_disp = coords[:, 1] - coords_ref[:, 1]  # Compute y displacement
                    x_disp = coords[:, 0] - coords_ref[:, 0]  # Compute x displacement
                    if y_disp.size == 0: raise ValueError("y_disp.size == 0")
                    if x_disp.size == 0: raise ValueError("x_disp.size == 0")
                    y_disp_mean = np.mean(y_disp)  # Calculate the mean displacement
                    x_disp_mean = np.mean(x_disp)  # Calculate the mean displacement
                    xy_displacements.append((x_disp_mean, y_disp_mean))  # Append the displacement to the list
                    break  # Exit the loop after finding the first match for this class_id
            else:
                # If no matching class_id_ref is found, append None (optional)
                xy_displacements.append((None,None))

        return xy_displacements
    
def all_frames_pix_displacement(tracking_files, reference_frames):
    # Get all x and y displacements
    all_xy_pix_displacement = [
        frame_pix_displacement(index, tracking_files, reference_frames) 
        for index, _ in enumerate(tracking_files)
    ]

    # Separate x and y displacements and compute their medians
    all_x_pix_median = [
        np.median([val[0] for val in frame_disp if val is not None and val[0] is not None]) 
        for frame_disp in all_xy_pix_displacement
    ]

    all_y_pix_median = [
        np.median([val[1] for val in frame_disp if val is not None and val[1] is not None]) 
        for frame_disp in all_xy_pix_displacement
    ]

    # Replace None values in all_xy_pix_displacement with the corresponding medians
    all_xy_pix_displacement_median = [
        [
            (
                val[0] if val is not None and val[0] is not None else all_x_pix_median[frame_index],
                val[1] if val is not None and val[1] is not None else all_y_pix_median[frame_index]
            )
            for val in frame_disp
        ]
        for frame_index, frame_disp in enumerate(all_xy_pix_displacement)
    ]

    return all_xy_pix_displacement_median

def remove_pix_displacement(tracking_folder, reference_frames):
    tracking_files = list(tracking_folder.iterdir())
    tracking_files.sort(key=lambda x: int(x.stem.split('_')[2]))
    pix_displacement = all_frames_pix_displacement(tracking_files, reference_frames)
    for track_path, frame_displacements in zip(tracking_files, pix_displacement):
        points = load_track_points(str(track_path))
        modified_points = []
        for (coords, metadata), displacement in zip(points, frame_displacements):
            coords[:, 0] -= displacement[0]
            coords[:, 1] -= displacement[1]  # Subtract pix_dist from the x and y-values
            modified_points.append((coords, metadata))  # Reconstruct the tuple
        save_track_points(str(track_path), modified_points)
    print("drone displacement accounted for")

def transform_trackbb_tostitch(base_folder, stitching_log):
    """
    Process the stitching iterations using the stitching log.

    Args:
        base_folder (str): Path to the base folder containing track files.
        stitching_log (list): Log of stitching operations with frame indices and transformation matrices.
    """
    for entry in stitching_log:
        operation_id = entry["operation_id"]
        input_frames = entry["input_frames"]
        result_frame = entry["result_frame"]
        transformation_matrices = entry["transformation_matrices"]

        print(f"Processing {operation_id}...")

        # Extract frame ranges and transformation matrices
        frame_a_indices = input_frames[0]["frame_indices"]
        frame_b_indices = input_frames[1]["frame_indices"]
        H, T, R, S = transformation_matrices

        # Process A points
        for frame in frame_a_indices:
            frame_path = f"{base_folder}/track_fr_{frame:04d}.txt"
            points = load_track_points(frame_path)

            # Transform all points at once using transform_first_points
            modified_points = [
                (transform_first_points(point_set, H, T, R, S), metadata)
                for point_set, metadata in points
            ]
            save_track_points(frame_path, modified_points)

        # Process B points
        for frame in frame_b_indices:
            frame_path = f"{base_folder}/track_fr_{frame:04d}.txt"
            points = load_track_points(frame_path)

            # Transform all points at once using transform_second_points
            modified_points = [
                (transform_second_points(point_set, H, T, R, S), metadata)
                for point_set, metadata in points
            ]
            save_track_points(frame_path, modified_points)

        print(f"  Completed processing for {operation_id}.")
