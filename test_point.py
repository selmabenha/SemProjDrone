from stitching_utils import *
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import numpy as np
import cv2
import gc
import os
import re
# import logging
from disp_utils import *

from PIL import Image


# logging.basicConfig(
#     level=logging.DEBUG,  # Set the minimum level of messages to capture
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("logs/script.log"),  # Write logs to this file
#         logging.StreamHandler()  # Optionally, also logging.info to console
#     ]
# )

print("Script started - test")

torch.set_grad_enabled(False)
images = Path("/home/finette/VideoStitching/selma/extracted_frames")
extract_frames_done = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = DoGHardNet(max_num_keypoints=None).eval().to(device)  # load the extractor
matcher = LightGlue(features="doghardnet").eval().to(device)
print("extractor and matcher done")

frame_step = 50
overlap_threshold = 2000  # Threshold for sufficient overlap
min_matches = 1000  # Threshold for minimal matches before there is a problem

output_images = "/home/finette/VideoStitching/selma/test_points"
if not os.path.exists(output_images):
    os.makedirs(output_images)

extract_frames_done = True






# image_paths = sorted(list(images.glob("*.jpg")))  # Adjust this to the path of your images

# def get_images_frames():
#     # Create a list of tuples (frame_number, image_path)
#     image_with_frame_numbers = []
#     images_cv = []
#     frame_ranges = []
#     # Extract frame numbers and file paths
#     for path in image_paths:
#         match = re.search(r'frame_(\d+)\.jpg', str(path))  # Match the frame number part
#         if match:
#             frame_number = int(match.group(1))  # Extract the frame number as an integer
#             image_with_frame_numbers.append((frame_number, path))

#     # Define the logic for frame ranges and add images to the list
#     for i, (frame_number, path) in enumerate(image_with_frame_numbers):
#         image = cv2.imread(str(path))

#         # Resize the image
#         height, width, _ = image.shape
#         reduced_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

#         # Determine the frame number range for the image
#         if i == len(image_with_frame_numbers) - 1:
#             frame_range = (frame_number, frame_number)  # Last image, map it to itself
#         else:
#             frame_range = (frame_number, image_with_frame_numbers[i + 1][0] - 1)  # Normal case

#         # Append the image and frame range to the respective lists
#         images_cv.append(reduced_image)
#         frame_ranges.append(frame_range)

#     return images_cv, frame_ranges


# images_cv = []
# i = 0
# for path in image_paths[12:14]:
#     i=i+1
#     image = cv2.imread(str(path))
#     height, width, _ = image.shape
#     reduced_image = cv2.resize(image, (width//2, height//2), interpolation=cv2.INTER_AREA)
#     images_cv.append(reduced_image)

# images_cv, frame_ranges = get_images_frames()
# print(f"len(images_cv) = {len(images_cv)}, len(frame_ranges) = {len(frame_ranges)}")
# print(f"frame_ranges = {frame_ranges}")


def transform_point_A_and_B_with_rotation(A, B, H, translation_matrix, rotation_matrix):
    # Transform point A (image0)
    A_homogeneous = np.array([[A]], dtype=np.float32)  # Shape (1, 1, 2)
    transformed_A = cv2.perspectiveTransform(A_homogeneous, H)  # Apply H

    # Create full 3x3 translation matrix
    full_translation_matrix = np.eye(3, dtype=np.float32)
    full_translation_matrix[:2, 2] = translation_matrix[:, 2]

    # Apply translation to transformed_A
    transformed_A_homogeneous = np.dot(full_translation_matrix, np.append(transformed_A[0][0], 1))
    final_A = transformed_A_homogeneous[:2] / transformed_A_homogeneous[2]

    # Transform point B (image1)
    # Step 1: Rotate point B
    rotated_B = np.dot(rotation_matrix, np.append(B, 1))[:2]  # Apply rotation matrix

    # Step 2: Translate rotated B
    B_homogeneous = np.append(rotated_B, 1)  # Convert to homogeneous coordinates
    transformed_B_homogeneous = np.dot(translation_matrix, B_homogeneous)  # Apply T
    final_B = transformed_B_homogeneous[:2]

    return final_A.astype(int), final_B.astype(int)




# A = (100, 200)
# B = (200, 100)

# cv2.circle(images_cv[0], A, 10, (0, 255, 0), -1)
# cv2.circle(images_cv[1], B, 10, (0, 255, 0), -1)
# image0 = images_cv[0]
# image1 = images_cv[1]

# # Get Transformed Points
# result_cv, transform_matrices = stitch_images_in_pairs(images_cv, True)
# logging.info(f"H = {H}")
# logging.info(f"translation_matrix = {translation_matrix}")
# logging.info(f"rotation_matrix = {rotation_matrix}")
# A_new, B_new = transform_point_A_and_B_with_rotation(A, B, H, translation_matrix, rotation_matrix)
# logging.info(f"A = {A}, A_new = {A_new}, B = {B}, B_new = {B_new}")

# cv2.circle(result_cv[0], A_new, 10, (255, 0, 0), -1)
# cv2.circle(result_cv[0], B_new, 10, (255, 0, 0), -1)

# display_original_merged(images_cv[0], images_cv[1], result_cv[0], 1, output_images)
