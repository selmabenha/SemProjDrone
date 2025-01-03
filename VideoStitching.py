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
# images = Path("/home/finette/VideoStitching/selma/extracted_frames")
# images = Path("/home/finette/VideoStitching/selma/output/images/trad_final")
images = Path("/home/finette/VideoStitching/selma/test/test_imgs")

output_images = "/home/finette/VideoStitching/selma/output/images"
if not os.path.exists(output_images):
    os.makedirs(output_images)


# COMMENT # COMMENT # COMMENT

image_paths = sorted(list(images.glob("*.jpg")))  # Adjust this to the path of your images

images_cv, frame_list = get_images_frames(image_paths, 1299)

# Traditional way
stitching_log = []
result_cv, frame_list, stitching_log = stitch_images_in_pairs(images_cv, frame_list, stitching_log, True)


logging.info(f"Amount of images at end of traditional way = {len(result_cv)}")
# Save all images in result_cv
for i, img_cv in enumerate(result_cv):
    output_filename = os.path.join(output_images, f"final_trad_{i}.jpg")
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


if stitching_log: 
    write_stitching_log(stitching_log)
    logging.info(f"Stitching log saved! {stitching_log}")







# # New Way
# result_cv = images_cv
# print(f"first stitching result, we are left with {len(result_cv)} images")
# # Main stitching loop
# crop_test = []
# transform_matrix = [0]
# prev_len = len(result_cv)  # Track progress by checking the length of the image list

# while True:
#     result_cv, crop_test_it, transform_matrix = stitch_images_pair_combos(result_cv, True)
#     crop_test.append(crop_test_it)

#     # Check if progress was made
#     current_len = len(result_cv)
#     if current_len == prev_len:
#         print("No further progress possible. Exiting loop.")
#         break  # Exit if no progress is made
#     prev_len = current_len  # Update previous length
#     # all_transform_matrices.append(transform_matrix)

# print(f"Stitch images pair combos with zoom, we are left with {len(result_cv)} images")
# print(f"crop test is = {crop_test}")

# # crop_test = []
# # transform_matrix = [0]
# # # Last ditch, check everything
# # while transform_matrix is not None:
# #     result_cv, crop_test_it, transform_matrix = stitch_images_pair_combos(result_cv, True)
# #     crop_test.append(crop_test_it)
# #     # all_transform_matrices.append(transform_matrix)

# # print(f"Stitch images pair combos #2, we are left with {len(result_cv)} images")
# # print(f"crop test is = {crop_test}")

if len(result_cv) == 1 or len(result_cv) >= 50:
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




