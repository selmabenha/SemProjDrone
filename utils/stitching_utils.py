from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet
from lightglue.utils import load_image, rbd
# from lightglue import viz2d
import torch
import numpy as np
import cv2
import gc
import os
import logging
from .disp_utils import *

from PIL import Image

min_matches = 1000  # Threshold for minimal matches before there is a problem

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = DoGHardNet(max_num_keypoints=None).eval().to(device)  # load the extractor
matcher = LightGlue(features="doghardnet").eval().to(device)


output_folder = "/Users/selmabenhassine/Desktop/SemProjDrone/output/images/base_out"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
### Revelant Functions

## EXTRACT MATCHES BETWEEN 2 IMAGES

# Filter the outer edges of each image to find matches where there is less
# likelihood of warping
def create_center_mask(image, center_fraction):
    # Get image dimensions
    h, w = image.shape[1:3]  # Height and width of the image

    # Calculate the center bounding box
    x_start = int((1 - center_fraction) * w / 2)
    x_end = int((1 + center_fraction) * w / 2)
    y_start = int((1 - center_fraction) * h / 2)
    y_end = int((1 + center_fraction) * h / 2)

    # Create a mask with zeros
    mask = torch.zeros_like(image)

    # Set the center region to 1
    mask[:, y_start:y_end, x_start:x_end] = 1

    return mask

# Modify the 'extract_matches' function to move images and tensors to the device
def extract_matches(image0, image1, center_filter, disp):
    # # Ensure input images are on the device
    image0 = image0.to(device)
    image1 = image1.to(device)

    # Create center masks for both images and move them to the device
    mask0 = create_center_mask(image0, center_filter).to(device)
    mask1 = create_center_mask(image1, center_filter).to(device)

    # Apply the masks to the images
    masked_image0 = image0 * mask0
    masked_image1 = image1 * mask1

    feats0 = extractor.extract(masked_image0.to(device))
    feats1 = extractor.extract(masked_image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Plotting - delete later?
    print(f"{len(m_kpts0)} <= min_matches and {disp}")
    if len(m_kpts0) <= min_matches and disp:
        display_extracted_matches(matches, image0, image1, m_kpts0, m_kpts1, kpts0, kpts1, matches01, output_folder)

    return m_kpts0, m_kpts1

# Rotate the image to find best matches
def rotate_image(image, angle):
    # Get the image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Generate rotation matrix and rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image, rotation_matrix

# Optimize number of matches by including rotation, which isn't factored in
# the tranformation
def find_best_rotation_matches(first_image, second_image, n, disp):
    best_matches0 = []
    best_rotated_image = None
    best_angle = 0
    best_rotation_matrix = []

    # Divide the full circle into `n` angles
    angle_step = 360 / n

    first_py = convert_cv_to_py(first_image)

    for i in range(n):
        # Calculate the current rotation angle
        angle = i * angle_step
        # Rotate the second image by this angle
        rotated_image, rotation_matrix = rotate_image(second_image.copy(), angle)

        # Extract matches between first image and rotated second image
        rotated_py = convert_cv_to_py(rotated_image)
        matches0, matches1 = extract_matches(first_py, rotated_py, 1, False)

        # If current rotation yields more matches, update best matches
        if matches0 is not None and len(matches0) > len(best_matches0):
          best_matches0 = matches0.clone()
          best_matches1 = matches1.clone()
          best_rotated_image = rotated_image.copy()
          best_rotated_image_py = rotated_py
          best_angle = angle
          best_rotation_matrix = rotation_matrix

    # Return the first image, best rotated version of the second image, and the best matches
    return first_image, best_rotated_image, best_matches0, best_matches1, best_angle, best_rotation_matrix

## TRANSFORMATION & STITCHING OF 2 IMAGES 
# COMMENT
def get_translation_matrix(all_corners):
    # Find bounding box of all corners
    x_min_, y_min_ = np.int32(all_corners.min(axis=0))
    x_max_, y_max_ = np.int32(all_corners.max(axis=0))

    # Calculate the size of the output image
    output_width_ = x_max_ - x_min_
    output_height_ = y_max_ - y_min_

    # Calculate the translation matrix to shift the image
    translation_matrix = np.float32([[1, 0, -x_min_], [0, 1, -y_min_]])  # 2x3 matrix

    return translation_matrix, output_width_, output_height_

def transform_images(image0_cv, image1_cv, matches0, matches1, disp):
    # Get image dimensions
    h1, w1 = image0_cv.shape[:2]
    h2, w2 = image1_cv.shape[:2]

    # Compute homography matrix H using OpenCV's RANSAC
    H, mask = cv2.findHomography(matches0.cpu().numpy(), matches1.cpu().numpy(), cv2.RANSAC)

    # Ensure H is of type float32
    H = H.astype(np.float32)

    # Corners of image 0
    corners0 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    # Corners of image 1
    corners1 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)

    # Transform corners of image 0
    transformed_corners0 = cv2.perspectiveTransform(corners0[None, :], H)[0]

    # Combine all corners
    all_corners = np.vstack((transformed_corners0, corners1))
    translation_matrix, output_width, output_height = get_translation_matrix(all_corners)

    # Create a full 3x3 translation matrix
    full_translation_matrix = np.eye(3, dtype=np.float32)
    full_translation_matrix[:2, 2] = translation_matrix[:, 2]  # Use only the translation part

    # Warp the images
    image0_transformed = cv2.warpPerspective(image0_cv, full_translation_matrix.dot(H), (output_width, output_height))
    image1_transformed = cv2.warpAffine(image1_cv, translation_matrix, (output_width, output_height))  # No need for [:2]

    length_matches = len(matches0)
    if disp == True and length_matches < min_matches:
        display_transformed_images(image0_transformed, image1_transformed, length_matches, output_folder)

    return image0_transformed, image1_transformed, H, translation_matrix

def merge_images(base_img, to_merge, device):
    base_img = convert_cv_to_py(base_img)
    to_merge = convert_cv_to_py(to_merge)
    # # Ensure inputs are PyTorch tensors
    # if isinstance(base_img, np.ndarray):
    #     base_img = torch.from_numpy(base_img).permute(2, 0, 1).float() / 255.0
    # if isinstance(to_merge, np.ndarray):
    #     to_merge = torch.from_numpy(to_merge).permute(2, 0, 1).float() / 255.0

    # Ensure tensors are on the specified device
    base_img = base_img.to(device)
    to_merge = to_merge.to(device)

    # Find all-zero locations in the base image
    all_zero_locs = torch.where(base_img == torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1).unsqueeze(-1))

    # Update the base image with the corresponding pixels from to_merge
    base_img[all_zero_locs[0], all_zero_locs[1], all_zero_locs[2]] = to_merge[all_zero_locs[0], all_zero_locs[1], all_zero_locs[2]]

    base_img = convert_py_to_cv(base_img)

    return base_img


# COMMENT # COMMENT # COMMENT

def convert_cv_to_py(image_cv):
    # Check if the input is a valid image
    if image_cv is None:
        raise ValueError("Received None for image_cv in convert_cv_to_py.")
    if not isinstance(image_cv, np.ndarray):
        logging.info(f"Expected np.ndarray, but got {type(image_cv)} in convert_cv_to_py.")
        return image_cv
    if image_cv.ndim != 3 or image_cv.shape[2] != 3:
        raise ValueError(f"Image should have 3 channels (RGB/BGR), got shape: {image_cv.shape}.")
    
    # Convert from BGR to RGB
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Convert to a PyTorch tensor
    image_py = torch.from_numpy(image_cv_rgb).permute(2, 0, 1)  # Change shape to [C, H, W]
    # Normalize the image to range [0, 1]
    image_py = image_py.float() / 255.0

    return image_py

def convert_py_to_cv(image_py):
    # Check if the input is a valid image
    if image_py is None:
        raise ValueError("Received None for image_py in convert_py_to_cv.")
    if not torch.is_tensor(image_py):
        logging.info(f"Expected tensor, but got {type(image_cv)} in convert_py_to_cv.")
        return image_py

    # Ensure the tensor is on CPU and detach if it's part of a computation graph
    image_np = image_py.cpu().numpy()

    # Convert from [C, H, W] to [H, W, C]
    image_np_hwc = np.transpose(image_np, (1, 2, 0))

    # Scale back to range [0, 255] and convert to uint8
    image_cv = (image_np_hwc * 255.0).astype(np.uint8)

    # Convert from RGB to BGR (OpenCV default format)
    image_cv_bgr = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    return image_cv_bgr


# COMMENT # COMMENT # COMMENT

def stitch_two_images(first_cv, second_cv, disp):
    # # Extract matches
    # first_py = convert_cv_to_py(first_cv)
    # second_py = convert_cv_to_py(second_cv)
    first_cv, second_cv, matches0, matches1, best_angle, R = find_best_rotation_matches(first_cv, second_cv, 20, True)
    if matches0 is None or len(matches0) < min_matches:
        logging.info(f"Not enough matches found, skipping. len(matches0) = {len(matches0)}")
        return first_cv, len(matches0), []
    else:
        logging.info(f"Best angle: {best_angle} and amount of matches: {len(matches0)}")

    # Transform images and stitch together
    first_transformed, second_transformed, H, T = transform_images(first_cv, second_cv, matches0, matches1, disp)
    if np.isnan(H).any() or np.isinf(H).any():
        logging.info(f"Invalid homography matrix, skipping.")
        return first_cv

    # Merge
    result_cv = merge_images(first_transformed, second_transformed, device)

    # Convert matrices to lists before saving
    H_list = H.tolist() if isinstance(H, np.ndarray) else H
    T_list = T.tolist() if isinstance(T, np.ndarray) else T
    R_list = R.tolist() if isinstance(R, np.ndarray) else R
    transform_matrices = [H_list, T_list, R_list]

    return result_cv, len(matches0), transform_matrices

# COMMENT # COMMENT # COMMENT

def stitch_images_in_pairs(image_list, frames_list, disp):
    if len(image_list) < 2:
        logging.info("Not enough images to stitch.")
        return image_list, frames_list, None

    current_images = image_list
    current_frames_list = frames_list
    full_frames_list = [current_frames_list]
    all_transform_matrices = []

    while len(current_images) > 1:
        logging.info(f"Stitching {len(current_images)} images...")
        next_round = []
        next_frames_list = []
        current_round_transform = []

        i = 0
        while i < len(current_images):
            if i + 1 < len(current_images):
                logging.info(f"Stitching images {i} and {i+1}")
                stitched_image, num_matches, current_stitch_transform = stitch_two_images(current_images[i], current_images[i+1], disp)

                if disp and num_matches <= min_matches:
                    display_original_merged(current_images[i], current_images[i+1], stitched_image, i, output_folder)
                    logging.info(f"Too few number of matches, stopped stitching: {num_matches}")
                    return current_images, all_transform_matrices, full_frames_list
                else:
                    next_round.append(stitched_image)
                    current_round_transform.append(current_stitch_transform)

                    # Merge frame ranges
                    new_frame_range = (current_frames_list[i][0], current_frames_list[i+1][1])
                    next_frames_list.append(new_frame_range)

                i += 2
            else:
                logging.info(f"Carrying forward the last image {i}")
                next_round.append(current_images[i])
                next_frames_list.append(current_frames_list[i])  # Carry forward its range
                i += 1

        all_transform_matrices.append(current_round_transform)
        full_frames_list.append(next_frames_list)
        current_images = next_round
        current_frames_list = next_frames_list

    return current_images, all_transform_matrices, full_frames_list


def stitch_images_pair_combos(image_list, disp): 
    crop_test = []
    if len(image_list) < 2:
        logging.info("Not enough images to stitch.")
        return image_list, None
    
    for i in range(len(image_list) - 1):
        for j in range(i + 1, len(image_list)):  # Ensure we don't repeat pairs
            # Stitch two images together
            _, _, matches0, matches1, _, _ = find_best_rotation_matches(image_list[i], image_list[j], 20, disp)
            num_matches = len(matches0)
            if num_matches >= min_matches:
                stitched_image, num_matches, current_stitch_transform = stitch_two_images(image_list[i], image_list[j], disp)
                print(f"Success! Stitched images {i} and {j}, with {num_matches} matches.")
                # Return the new image list and transformation matrix
                new_image_list = [image_list[k] for k in range(len(image_list)) if k != i and k != j] + [stitched_image]
                return new_image_list, current_stitch_transform
            else:
                new_matches0, new_matches1 = zoom_and_test(image_list[i], image_list[j], matches0, matches1, True)
                crop_test.append([i, j, len(matches0), len(new_matches0)])
                print(f"Combo {i}, {j}: {num_matches} matches. Too few matches.")

    print("no combos work, sorry")
    return image_list, crop_test

def get_bounding_box(matches0, matches1, margin=20):
    """
    Given the matches, return the bounding box around the area with the dense matches.
    The margin is added around the bounding box to ensure enough context around the matches.
    """
    # Extract keypoint coordinates (assuming they are in the form (x, y))
    coords0 = np.array([match.pt for match in matches0])
    coords1 = np.array([match.pt for match in matches1])
    
    # Find the min/max coordinates to define a bounding box around the matching points
    min_x0, min_y0 = np.min(coords0, axis=0)
    max_x0, max_y0 = np.max(coords0, axis=0)
    min_x1, min_y1 = np.min(coords1, axis=0)
    max_x1, max_y1 = np.max(coords1, axis=0)
    
    # Apply margin to the bounding box
    min_x0 = max(min_x0 - margin, 0)  # Ensure we don't go out of image bounds
    min_y0 = max(min_y0 - margin, 0)
    max_x0 = min(max_x0 + margin, matches0[0].image.shape[1])  # Limit to image width
    max_y0 = min(max_y0 + margin, matches0[0].image.shape[0])  # Limit to image height
    
    min_x1 = max(min_x1 - margin, 0)
    min_y1 = max(min_y1 - margin, 0)
    max_x1 = min(max_x1 + margin, matches1[0].image.shape[1])
    max_y1 = min(max_y1 + margin, matches1[0].image.shape[0])
    
    return (min_x0, min_y0, max_x0, max_y0), (min_x1, min_y1, max_x1, max_y1)

def zoom_and_test(image0_cv, image1_cv, matches0, matches1, disp, margin=20):
    # Get the bounding boxes for both images based on matches with margin
    bbox0, bbox1 = get_bounding_box(matches0, matches1, margin)
    
    # Crop the images based on the bounding box to zoom in on the matching area
    cropped_image0 = image0_cv[int(bbox0[1]):int(bbox0[3]), int(bbox0[0]):int(bbox0[2])]
    cropped_image1 = image1_cv[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
    
    # Recalculate the matches for the cropped region
    m_kpts0, m_kpts1 = extract_matches(cropped_image0, cropped_image1, center_filter=True, disp=disp)

    return m_kpts0, m_kpts1