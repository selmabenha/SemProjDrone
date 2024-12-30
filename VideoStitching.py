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
images = Path("/home/finette/VideoStitching/selma/extracted_frames")

output_images = "/home/finette/VideoStitching/selma/output/images/base_out"
if not os.path.exists(output_images):
    os.makedirs(output_images)


# COMMENT # COMMENT # COMMENT

image_paths = sorted(list(images.glob("*.jpg")))  # Adjust this to the path of your images

images_cv, frame_ranges = get_images_frames(image_paths)

# Traditional way
result_cv, all_transform_matrices, full_frames_recording = stitch_images_in_pairs(images_cv, frame_ranges, True)

transform_matrix = []
# Last ditch, check everything
while transform_matrix is not None:
    result_cv, transform_matrix = stitch_images_pair_combos(result_cv, True)
    all_transform_matrices.append(transform_matrix)

# i = 0
# while len(result_cv) != 1 and i < 3:
#     logging.info(f"retry stitching #{i}")
#     for img in result_cv:
#         logging.info(f"Height: {img.shape[0]}, Width: {img.shape[1]}")

#     result_cv = sorted(result_cv, key=lambda img: img.shape[0] * img.shape[1])
#     result_cv, new_transform_matrices, full_frames_recording = stitch_images_in_pairs(result_cv, full_frames_recording, True)
#     if not (new_transform_matrices and any(new_transform_matrices)): 
#         all_transform_matrices.append(new_transform_matrices)


#     print(f"Check stitching for iteration {i}")
#     check_results_type(result_cv)

#     i += 1

if all_transform_matrices is not None: 
    write_images_frames(full_frames_recording, all_transform_matrices)
    logging.info(f"All transform matrices = {all_transform_matrices}")
    logging.info(f"full_frames_recording = {full_frames_recording}")

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
    # Save all images in result_cv
    for i, img_cv in enumerate(result_cv):
        output_filename = os.path.join(output_images, f"final_{i}.jpg")
        try:
            if not cv2.imwrite(output_filename, img_cv):
                logging.info(f"Error saving image {output_filename} using OpenCV.")
                try:
                    img = Image.fromarray(img_cv)
                    img.save(output_filename)
                    logging.info(f"Image {i} saved successfully to {output_filename}")
                except Exception as e:
                    logging.info(f"Error saving image {output_filename} with PIL: {e}")
            else:
                logging.info(f"Image {i} saved successfully to {output_filename} using OpenCV")
        except Exception as e:
            logging.info(f"Unexpected error saving image {output_filename}: {e}")




