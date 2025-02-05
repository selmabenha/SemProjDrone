from pathlib import Path
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.set_grad_enabled(False)
images = Path("extracted_frames")

output_images = "output/images"
if not os.path.exists(output_images):
    os.makedirs(output_images)


image_paths = sorted(list(images.glob("*.jpg")))  # Adjust this to the path of your images

images_cv, frame_list = get_images_frames(image_paths, 7517)
stitching_log = []
i = 0
result_cv, frame_list, stitching_log = stitch_images_in_pairs(images_cv, frame_list, stitching_log, True)

save_intermediate_images(result_cv, output_images, i)

logging.info(f"Amount of images at end of traditional way = {len(result_cv)}")

if stitching_log: 
    write_stitching_log(stitching_log)
    logging.info("Stitching log saved! ")

torch.cuda.empty_cache()
gc.collect()

# Last ditch effort - try different combinations

logging.info(f"FIRST STITCHING RESULT, we are left with {len(result_cv)} images")

crop_test = []
transform_matrix = [0]
prev_len = len(result_cv)  # Track progress by checking the length of the image list
i = 1
while True:
    logging.info(f"Iteration {i}")
    new_result_cv, crop_test_it, _ = stitch_images_pair_combos(result_cv, True)
    crop_test.append(crop_test_it)
    save_intermediate_images(new_result_cv, output_images, i)
    i+=1

    # Check if progress was made
    current_len = len(new_result_cv)
    if current_len == prev_len:
        logging.info("No further progress possible. Exiting loop.")
        result_cv = new_result_cv
        break  # Exit if no progress is made
    else: 
        logging.info("success, something was done. Try again!")
        prev_len = current_len  # Update previous length
        result_cv = new_result_cv