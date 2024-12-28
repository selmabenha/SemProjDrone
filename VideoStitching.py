# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

# if Path.cwd().name != "LightGlue":
#     !git clone --quiet https://github.com/cvg/LightGlue/
#     %cd LightGlue
#     !pip install --progress-bar off --quiet -e .

from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import numpy as np
import cv2
import gc
import os
import logging
from utils import *
import re

from PIL import Image

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

logging.info("Script started - main")


torch.set_grad_enabled(False)
images = Path("/home/finette/VideoStitching/selma/output/extracted_frames")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
# extractor = DoGHardNet(max_num_keypoints=None).eval().to(device)  # load the extractor
# matcher = LightGlue(features="doghardnet").eval().to(device)
# logging.info("extractor and matcher done")
# Path to video file
# video_path = "selma/DJI_0763.MOV"
output_folder = "selma/output_frames"
frame_step = 50
overlap_threshold = 2000  # Threshold for sufficient overlap
min_matches = 1000  # Threshold for minimal matches before there is a problem

output_images = "/home/finette/VideoStitching/selma/output/images/test_transform"
if not os.path.exists(output_images):
    os.makedirs(output_images)


# COMMENT # COMMENT # COMMENT

image_paths = sorted(list(images.glob("*.jpg")))  # Adjust this to the path of your images


images_cv, frame_ranges = get_images_frames(image_paths)
print(f"len(images_cv) = {len(images_cv)}, len(frame_ranges) = {len(frame_ranges)}")
print(f"frame_ranges = {frame_ranges}")


result_cv, all_transform_matrices, full_frames_recording = stitch_images_in_pairs(images_cv, frame_ranges, True)
if all_transform_matrices is not None: 
    write_images_frames(full_frames_recording, all_transform_matrices)
    logging.info(f"All transform matrices = {all_transform_matrices}")
    logging.info(f"full_frames_recording = {full_frames_recording}")

i = 0
while len(result_cv) != 1 and i < 3:
    logging.info(f"retry stitching #{i}")
    result_cv = sorted(result_cv, key=lambda img: img.shape[0] * img.shape[1])
    for img in result_cv:
        logging.info(f"Height: {img.shape[0]}, Width: {img.shape[1]}")
    result_cv = stitch_images_in_pairs(result_cv, full_frames_recording, True)
    i += 1

if len(result_cv) == 1:
    result_cv = result_cv[0]
    # Assuming base_image_cv is your final stitched image
    height, width, _ = result_cv.shape  # Get image dimensions

    # Save the final stitched image instead of displaying it
    output_filename = os.path.join(output_images, "final_stitched_image.jpg")
    if not cv2.imwrite(output_filename, result_cv):
        logging.info(f"Error saving final stitched image to {output_filename}")
        try:
            img = Image.fromarray(result_cv)
            img.save(output_filename)
            logging.info(f"Image saved successfully to {output_filename}")
        except Exception as e:
            logging.info(f"Error saving image {output_filename}: {e}")
    else:
        logging.info(f"Saved final stitched image to {output_filename}")

else:
    logging.info(f"Final amount of images = {len(result_cv)}")
    display_transformed_images(result_cv[0], result_cv[1], 0)
    display_transformed_images(result_cv[2], result_cv[3], 1)




