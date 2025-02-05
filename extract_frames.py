from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import os
import logging
from utils import *

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

logging.info("Script started - extract")

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=None).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

# Path to video file
video_path = "DJI_0763.MOV"
output_folder = "extracted_frames"
frame_step = 50
overlap_threshold = 2000  # Threshold for sufficient overlap
min_matches = 500  # Threshold for minimal matches before there is a problem

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not video_capture.isOpened():
    logging.info("Error: Could not open video.")
    exit()

# Read the first frame to initialize
success, prev_frame = video_capture.read()
# If reading a frame was not successful, break the loop (end of video)
if not success:
    logging.info("Error: Could not open read.")
    exit()

frame_count = 0
saved_frame_count = 1
frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
cv2.imwrite(frame_filename, prev_frame)
frame_count = 1
matches_per_frame = []
# Loop through the video frames
while True:
    # Read the next frame from the video
    success, frame = video_capture.read()

    # If reading a frame was not successful, break the loop (end of video)
    if not success:
        break

    # Check if the current frame is one to save based on the adjusted frame_step
    if frame_count % frame_step == 0:
        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Get new and most recent image to compare matches
        image_paths = sorted(list(Path(output_folder).glob("*.jpg")))
        saved_cv = cv2.imread(str(image_paths[-2]))
        height, width, _ = saved_cv.shape
        saved_reduced = cv2.resize(saved_cv, (width//2, height//2), interpolation=cv2.INTER_AREA)

        new_cv = cv2.imread(str(image_paths[-1]))
        height, width, _ = new_cv.shape
        new_reduced = cv2.resize(new_cv, (width//2, height//2), interpolation=cv2.INTER_AREA)


        _, _, matches0, matches1, best_angle, R = find_best_rotation_matches(saved_reduced, new_reduced, 20, True)

        if len(matches0) <= overlap_threshold and len(matches0) >= min_matches:
            matches_per_frame.append([matches0, matches1, best_angle])
            saved_frame_count += 1
        else:
            if len(matches0) < min_matches: print(f"remove frame {frame_count}, {len(matches0)} matches!")
            try:
                os.remove(frame_filename)
            except Exception as e:
                logging.info(f"Error deleting file: {e}")

    # Increment the frame counter
    frame_count += 1

# Release the video capture object
video_capture.release()

logging.info(f"Extracted {saved_frame_count} frames from {video_path} and saved them to {output_folder}.")