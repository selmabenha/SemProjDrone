from pathlib import Path
import torch
import cv2
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

logging.info("Script started - transform mini all")

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = DoGHardNet(max_num_keypoints=None).eval().to(device)  # load the extractor
matcher = LightGlue(features="doghardnet").eval().to(device)

# Path to video file
video_path = "/home/finette/VideoStitching/ShortMovieVersion.mov"
output_folder = "/home/finette/VideoStitching/selma/output/extracted_frames"
track_folder = "/home/finette/VideoStitching/transform_small"
frame_step = 50
overlap_threshold = 2000  # Threshold for sufficient overlap
min_matches = 100  # Threshold for minimal matches before there is a problem

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not video_capture.isOpened():
    logging.info("Error: Could not open video.")
    exit()

frame_count = 0
# Read the first frame to initialize
success, prev_frame = video_capture.read()
# If reading a frame was not successful, break the loop (end of video)
if not success:
    logging.info("Error: Could not open read.")
    exit()

frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
cv2.imwrite(frame_filename, prev_frame)
frame_count = 1
saved_frame_count = 1

matches_per_frame = []
# Loop through the video frames
while True:
    # Read the next frame from the video
    success, frame = video_capture.read()

    # If reading a frame was not successful, break the loop (end of video)
    if not success:
        break

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


    # Use the device to process keypoints and matches
    _, _, matches0, matches1, best_angle, R = find_best_rotation_matches(saved_reduced, new_reduced, 20, True)
    _, _, H, T = transform_images(saved_reduced, new_reduced, matches0, matches1, True)

    # Conversion of frame A 
    frame_path_A = f"{track_folder}/track_fr_{(frame_count-1):04d}.txt"
    points_A = load_track_points(frame_path_A)

    # Transform each point as A
    modified_points_A = [
        (np.array([transform_point_A(pt, H, T, R) for pt in point_set]), metadata)
        for point_set, metadata in points_A
    ]
    save_track_points(frame_path_A, modified_points_A)

    frame_path_B = f"{track_folder}/track_fr_{frame_count:04d}.txt"
    points_B = load_track_points(frame_path_B)

    # Transform each point as B
    modified_points_B = [
        (np.array([transform_point_B(pt, H, T, R) for pt in point_set]), metadata)
        for point_set, metadata in points_B
    ]
    save_track_points(frame_path_B, modified_points_B)

    if len(matches0) <= overlap_threshold and len(matches0) >= min_matches and frame_count % frame_step == 0:
        matches_per_frame.append([matches0, matches1, best_angle])
        saved_frame_count += 1
    else:
        try:
            os.remove(frame_filename)
        except Exception as e:
            logging.info(f"Error deleting file: {e}")

    # Increment the frame counter
    frame_count += 1

# Release the video capture object
video_capture.release()

logging.info(f"Extracted {saved_frame_count} frames from {video_path} and saved them to {output_folder}.")
