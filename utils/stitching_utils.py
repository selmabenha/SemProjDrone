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
import torch.nn.functional as F
from utils.files_utils import *
from utils.disp_utils import *

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


output_folder = "output/images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
### Revelant Functions

## EXTRACT MATCHES BETWEEN 2 IMAGES

# Filter the outer edges of each image to find matches where there is less
# likelihood of warping
def create_center_mask(image, center_fraction):
    """
    Create a binary center mask for a given image and center fraction.
    
    Args:
        image (torch.Tensor): The input image tensor of shape (C, H, W).
        center_fraction (float): Fraction of the image (0 < center_fraction <= 1) to be included in the mask.

    Returns:
        torch.Tensor: A mask tensor of the same shape as the input image, with the center region set to 1.
    """
    # Get image dimensions
    h, w = image.shape[1:3]  # Height and width of the image

    # Calculate the center bounding box
    x_start = int((1 - center_fraction) * w / 2)
    x_end = int((1 + center_fraction) * w / 2)
    y_start = int((1 - center_fraction) * h / 2)
    y_end = int((1 + center_fraction) * h / 2)

    # Create the mask
    mask = torch.zeros((image.size(0), h, w), device=image.device, dtype=image.dtype)  # Create directly on the correct device
    mask[:, y_start:y_end, x_start:x_end] = 1  # Set the center region to 1

    # Explicitly release variables to free memory
    del h, w, x_start, x_end, y_start, y_end

    # Optionally, run garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    return mask


# Modify the 'extract_matches' function to move images and tensors to the device
def extract_matches(image0, image1, center_filter, disp=False):
    """
    Extract keypoint matches between two images with optional center filtering.

    Args:
        image0, image1: Input images (assumed to be in OpenCV format).
        center_filter: Fraction of the image to retain in the center.
        disp: Display flag for debugging matches.

    Returns:
        m_kpts0, m_kpts1: Matched keypoints from the two images.
    """
    feats0, feats1, matches01, kpts0, kpts1, matches = None, None, None, None, None, None  # Initialize variables

    try:
        # Convert and move images to device
        image0 = convert_cv_to_py(image0).to(device, non_blocking=True)
        image1 = convert_cv_to_py(image1).to(device, non_blocking=True)

        # Create and apply center masks
        mask0 = create_center_mask(image0, center_filter)  # Already on the device
        mask1 = create_center_mask(image1, center_filter)  # Already on the device
        masked_image0 = image0 * mask0
        masked_image1 = image1 * mask1

        # Check if masked images are empty
        if torch.count_nonzero(masked_image0) == 0 or torch.count_nonzero(masked_image1) == 0:
            raise ValueError("One or both masked images are empty or invalid")

        # Extract features
        feats0 = extractor.extract(masked_image0)
        feats1 = extractor.extract(masked_image1)

        max_features = 11000

        print("gonna slice now")

        # Apply slicing to the relevant fields (keypoints, scales, oris, descriptors, keypoint_scores)
        feats0 = {
            key: value[:, :max_features] if key in ["keypoints", "scales", "oris", "descriptors", "keypoint_scores"] else value
            for key, value in feats0.items()
        }

        feats1 = {
            key: value[:, :max_features] if key in ["keypoints", "scales", "oris", "descriptors", "keypoint_scores"] else value
            for key, value in feats1.items()
        }


        del masked_image0, masked_image1
        gc.collect()
        torch.cuda.empty_cache()


        # Match features
        matches01 = matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # Remove batch dimension

        # Extract keypoints and matches
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        del feats0, feats1, matches01
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        # logging.info debug information on error
        logging.info(f"Masked Image 0 - Shape: {image0.shape}, Type: {type(image0)}")
        logging.info(f"Masked Image 1 - Shape: {image1.shape}, Type: {type(image1)}")
        logging.info(f"ERROR: {e}")
        m_kpts0, m_kpts1 = None, None

    finally:
        # Cleanup to release CUDA memory
        del image0, image1, mask0, mask1
        del kpts0, kpts1, matches
        gc.collect()
        torch.cuda.empty_cache()
        

    return m_kpts0, m_kpts1


def rotate_image(image, angle):
    """
    Rotate the image by a specified angle and crop it to the bounding box of non-black areas.

    Args:
        image: Input image (NumPy array).
        angle: Rotation angle in degrees.

    Returns:
        cropped_image: Cropped rotated image.
        rotation_matrix: The rotation matrix used for transformation.
    """
    rotated_image = None
    try:
        # Get the image dimensions
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Compute the bounding box of the rotated image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])

        # Compute new bounding dimensions
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for the new dimensions
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

        cropped_image = crop_black_regions(rotated_image)

    except Exception as e:
        logging.info(f"Error rotating image: {e}")
        cropped_image = image
        rotation_matrix = None

    finally:
        # Explicit cleanup of variables to reduce memory usage
        del h, w, center, cos, sin, new_h, new_w, rotated_image
        gc.collect()
        torch.cuda.empty_cache()

    return cropped_image, rotation_matrix


def find_best_rotation_matches(first_image, second_image, n, disp, device="cuda"):
    logging.info("rotation")
    best_matches0, best_matches1 = [], []
    best_rotated_image = None
    best_angle = 0
    best_rotation_matrix = []

    # Divide the full circle into `n` angles
    angle_step = 360 / n

    # Convert the first image to PyTorch tensor and move it to the GPU
    first_py = convert_cv_to_py(first_image).to(device)

    for i in range(n):
        # Calculate the current rotation angle
        angle = i * angle_step
        
        # Rotate the second image using PyTorch
        rotated_image, rotation_matrix = rotate_image(second_image, angle)
        rotated_py = convert_cv_to_py(rotated_image).to(device)
        # Extract matches between the first image and the rotated second image
        matches0, matches1 = extract_matches(first_py, rotated_py, 1, False)

        # If the current rotation yields more matches, update the best matches
        if matches0 is not None and len(matches0) > len(best_matches0):
            best_matches0 = matches0
            best_matches1 = matches1
            best_rotated_image = rotated_image
            best_angle = angle
            best_rotation_matrix = rotation_matrix

        # Release memory for temporary variables
        del angle, rotation_matrix, rotated_py, matches0, matches1, rotated_image
        gc.collect()
        torch.cuda.empty_cache()


    del first_py
    gc.collect()
    torch.cuda.empty_cache()

    # Return the first image, best rotated version of the second image, and the best matches
    return first_image, best_rotated_image, best_matches0, best_matches1, best_angle, best_rotation_matrix

## TRANSFORMATION & STITCHING OF 2 IMAGES 
# COMMENT
def get_translation_matrix(all_corners, device="cuda"):
    # Ensure input is a PyTorch tensor
    all_corners = torch.as_tensor(all_corners, device=device, dtype=torch.float32)

    # Find the bounding box of all corners
    x_min_, y_min_ = torch.floor(all_corners.min(dim=0).values).int()
    x_max_, y_max_ = torch.ceil(all_corners.max(dim=0).values).int()

    # Calculate the size of the output image
    output_width_ = x_max_ - x_min_
    output_height_ = y_max_ - y_min_

    # Calculate the translation matrix to shift the image
    translation_matrix = torch.tensor(
        [[1, 0, -x_min_.item()], [0, 1, -y_min_.item()]],
        dtype=torch.float32,
        device=device,
    )

    # Release intermediate variables to free memory
    del x_min_, y_min_, x_max_, y_max_
    gc.collect()
    torch.cuda.empty_cache()


    return translation_matrix, output_width_.item(), output_height_.item()

def transform_images(first, second, matches0, matches1, disp=False, device="cuda"):
    image0_cv = crop_black_regions(first)
    image1_cv = crop_black_regions(second)

    # Ensure inputs are on the device
    image0_cv = torch.as_tensor(image0_cv, device=device, dtype=torch.float32)
    image1_cv = torch.as_tensor(image1_cv, device=device, dtype=torch.float32)
    matches0 = matches0.to(device)
    matches1 = matches1.to(device)

    # Get image dimensions (no need to create new variables, reuse existing)
    h1, w1 = image0_cv.shape[:2]
    h2, w2 = image1_cv.shape[:2]

    # Move matches to CPU and convert to NumPy arrays for OpenCV
    matches0_cpu = matches0.cpu().numpy()
    matches1_cpu = matches1.cpu().numpy()

    # Compute homography matrix H using OpenCV's RANSAC
    H, _ = cv2.findHomography(matches0_cpu, matches1_cpu, cv2.RANSAC)

    # Ensure H is of type float32
    H = H.astype(np.float32)

    if H[2, 2] != 0:  # Avoid division by zero
        H /= H[2, 2]
    H[2, 0] /= 2
    H[2, 1] /= 2

    # Corners of image 0
    corners0 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    # Corners of image 1
    corners1 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)

    # Transform corners of image 0
    transformed_corners0 = cv2.perspectiveTransform(corners0[None, :], H)[0]
    del corners0  # delete corners0 after use

    # Combine all corners
    all_corners = np.vstack((transformed_corners0, corners1))
    translation_matrix, output_width, output_height = get_translation_matrix(all_corners)
    del transformed_corners0, corners1, all_corners

    scale = [[output_width / w1, output_height / h1], [output_width / w2, output_height / h2]]

    # Create a full 3x3 translation matrix
    full_translation_matrix = np.eye(3, dtype=np.float32)

    # Move translation_matrix to CPU and convert to NumPy
    translation_matrix_cpu = translation_matrix.cpu().numpy()

    # Now assign the translation part to the full translation matrix
    full_translation_matrix[:2, 2] = translation_matrix_cpu[:, 2]  # Use only the translation part

    # Warp the images
    image0_transformed = cv2.warpPerspective(image0_cv.cpu().numpy(), full_translation_matrix.dot(H), (output_width, output_height))
    image1_transformed = cv2.warpAffine(image1_cv.cpu().numpy(), translation_matrix_cpu, (output_width, output_height))  # No need for [:2]

    # Clear memory for the warped images
    del image0_cv, image1_cv

    # Explicitly clear any remaining unused variables
    del full_translation_matrix, output_width, output_height

    # Free CUDA memory (if running on GPU)
    gc.collect()
    torch.cuda.empty_cache()

    return image0_transformed, image1_transformed, H, translation_matrix_cpu, scale

def rotate_and_crop(base_img, num_rotations=20, min_crop_percent=0.1, device='cuda'):
    """
    Rotates the image several times, crops black regions, and returns the image with the smallest non-black bounding box.
    
    Args:
        base_img: The original image (NumPy array).
        num_rotations: The number of rotations to try.
        min_crop_percent: Minimum percentage of the original image to retain when cropping.
        device: Device ('cuda' or 'cpu') to perform the computation on.
        
    Returns:
        The rotated image with the smallest bounding box of non-black regions.
    """
    best_image = base_img
    best_size = base_img.shape[0] * base_img.shape[1]  # Initial size is the area of the image

    # Rotate and crop the image at different angles
    for angle in np.linspace(0, 360, num_rotations, endpoint=False):
        # Rotate the image and crop it
        cropped_img, _ = rotate_image(base_img, angle)
        
        # Crop black regions
        cropped_img = crop_black_regions(cropped_img, min_crop_percent)

        # Check if this cropped image has a smaller size (less black area)
        cropped_size = cropped_img.shape[0] * cropped_img.shape[1]
        if cropped_size < best_size:
            best_image = cropped_img
            best_size = cropped_size

        # Clear GPU memory after each iteration to avoid fragmentation
        gc.collect()
        torch.cuda.empty_cache()

    return best_image

def merge_images(base_img, to_merge, device):
    # Convert the base image and to_merge to PyTorch tensors if they are in OpenCV format
    base_img = convert_cv_to_py(base_img).to(device)
    to_merge = convert_cv_to_py(to_merge).to(device)

    # Find all-zero locations in the base image
    all_zero_locs = torch.where(base_img == torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(-1).unsqueeze(-1))

    # Update the base image with the corresponding pixels from to_merge
    base_img[all_zero_locs[0], all_zero_locs[1], all_zero_locs[2]] = to_merge[all_zero_locs[0], all_zero_locs[1], all_zero_locs[2]]

    del to_merge, all_zero_locs
    gc.collect()
    torch.cuda.empty_cache()
    # Convert the modified tensor back to OpenCV format
    base_img = convert_py_to_cv(base_img)
    cropped_img = rotate_and_crop(base_img)

    # Free memory for tensors and intermediate variables no longer needed
    del base_img

    # Explicitly clear CUDA cache if needed
    gc.collect()
    torch.cuda.empty_cache()

    return cropped_img


# COMMENT # COMMENT # COMMENT
def convert_cv_to_py(image_cv):
    # Check if the input is a valid image
    if image_cv is None:
        raise ValueError("Received None for image_cv in convert_cv_to_py.")

    if isinstance(image_cv, np.ndarray):
        # Check dimensions and convert only if necessary
        if image_cv.ndim != 3 or image_cv.shape[2] != 3:
            raise ValueError(f"Image should have 3 channels (RGB/BGR), got shape: {image_cv.shape}.")
        # Convert BGR to RGB in-place to avoid a copy
        image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        # Directly convert to tensor and normalize in one step to reduce memory usage
        return torch.from_numpy(image_cv_rgb).permute(2, 0, 1).float().div(255.0)  # div(255.0) is more efficient than dividing in the conversion

    elif isinstance(image_cv, torch.Tensor):
        # Handle tensor input
        if len(image_cv.shape) == 3:
            # Check if tensor is in [C, H, W] or [H, W, C] format
            if image_cv.shape[0] == 3:  # Already in [C, H, W]
                return image_cv
            elif image_cv.shape[-1] == 3:  # Convert from [H, W, C] to [C, H, W]
                return image_cv.permute(2, 0, 1)
            else:
                raise ValueError(f"Invalid tensor shape for image: {image_cv.shape}")
        else:
            raise ValueError(f"Unexpected tensor dimensions: {image_cv.shape}")
    else:
        raise ValueError(f"Unexpected input type: {type(image_cv)}")


def convert_py_to_cv(image_py):
    # Check if the input is a valid image
    if image_py is None:
        raise ValueError("Received None for image_py in convert_py_to_cv.")

    if not torch.is_tensor(image_py):
        return image_py

    # Ensure the tensor is on CPU and detach if it's part of a computation graph
    image_py = image_py.cpu().detach()

    # Convert tensor to numpy directly and transpose to HWC format
    image_np_hwc = image_py.permute(1, 2, 0).numpy()

    # Scale back to range [0, 255] and convert to uint8 in one step
    image_cv_bgr = cv2.cvtColor((image_np_hwc * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    del image_np_hwc  # Only delete the large intermediate variable to free memory
    return image_cv_bgr


# COMMENT # COMMENT # COMMENT

def stitch_two_images(first, second, disp):
    # Make sure black regions are cropped out
    first_cv = crop_black_regions(first)
    second_cv = crop_black_regions(second)

    # Extract matches and handle early return if not enough matches
    first_cv, second_cv, matches0, matches1, best_angle, R = find_best_rotation_matches(first_cv, second_cv, 20, False)
    
    num_matches = len(matches0)
    if matches0 is None or num_matches < min_matches:
        logging.info(f"Not enough matches found, skipping. len(matches0) = {num_matches}")
        del first_cv, second_cv, matches0, matches1, best_angle, R  # Clear memory
        return None, num_matches, None

    logging.info(f"Best angle: {best_angle} and amount of matches: {num_matches}")

    # Transform images and stitch together, handle invalid homography matrix
    first_transformed, second_transformed, H, T, S = transform_images(first_cv, second_cv, matches0, matches1, disp)
    if np.isnan(H).any() or np.isinf(H).any():
        logging.info(f"Invalid homography matrix, skipping.")
        del first_cv, second_cv, matches0, matches1, best_angle, R, first_transformed, second_transformed, H, T, S
        return None, None, None

    # Merge images
    result_cv = merge_images(first_transformed, second_transformed, device)

    # Convert matrices to lists and prepare for return
    transform_matrices = [
        H.tolist() if isinstance(H, np.ndarray) else H,
        T.tolist() if isinstance(T, np.ndarray) else T,
        R.tolist() if isinstance(R, np.ndarray) else R,
        S.tolist() if isinstance(S, np.ndarray) else S
    ]

    # Clear all large variables from memory after use
    del first_cv, second_cv, matches0, matches1, best_angle, R, first_transformed, second_transformed, H, T, S

    return result_cv, num_matches, transform_matrices


# COMMENT # COMMENT # COMMENT

def stitch_images_in_pairs(image_list, frames_list, stitching_log, disp):
    if len(image_list) < 2:
        logging.info("Not enough images to stitch.")
        return image_list, frames_list, stitching_log

    operation_counter = 1  # Initialize operation counter for operation IDs
    iteration = 1  # Track iteration number for debugging

    prev_len_images = -1
    while len(image_list) != prev_len_images:
        prev_len_images = len(image_list)
        logging.info(f"Iteration {iteration}: Stitching {len(image_list)} images...")
        
        next_round = []
        next_frames_list = []

        i = 0
        while i < len(image_list):
            gc.collect()
            torch.cuda.empty_cache()

            if i + 1 < len(image_list):
                logging.info(f"  Stitching images {i} and {i+1}")
                
                # Perform stitching
                stitched_image, num_matches, current_stitch_transform = stitch_two_images(image_list[i], image_list[i+1], disp)

                if num_matches is None or num_matches < min_matches:
                    logging.info(f"Too few matches, {num_matches}, trying zoom.")
                    del stitched_image, num_matches, current_stitch_transform
                    _, _, best_matches0, best_matches1, _, _ = find_best_rotation_matches(image_list[i], image_list[i+1], 20, disp)
                    zoom_result, zoom_matches, zoom_frames_list, zoom_stitching_log = zoom_and_stitch(
                        image_list[i], image_list[i+1], best_matches0, best_matches1, True
                    )
                    del best_matches0, best_matches1
                    if zoom_matches < min_matches:
                        logging.info(f"Too few zoom matches, {zoom_matches}, trying split.")
                        del zoom_result, zoom_matches

                        split_result, split_matches, _, _ = split_and_stitch(image_list[i], image_list[i+1], True)
                        if split_matches < min_matches:
                            logging.info(f"  Too few split matches, {split_matches}, done stitching.")
                            del split_result, split_matches
                            return image_list, frames_list, stitching_log
                            # next_round.append(image_list[i])
                            # next_round.append(image_list[i+1])
                        else:
                            next_round.append(split_result)
                            if isinstance(split_result, list):
                                logging.info("split_result is a list.")
                                logging.info(f"Number of elements in the list: {len(split_result)}")
                                logging.info(f"Types of elements in the list: {[type(item) for item in split_result]}")
                            del split_result, split_matches
                            operation_id = f"operation_{operation_counter:03d}"
                            logging.info(f"  Logged operation {operation_id}") #: Frames {new_frame_range['indices']}")
                            operation_counter += 1
                    else:
                        if isinstance(zoom_result, list):
                            logging.info("zoom_result is a list.")
                            logging.info(f"Number of elements in the list: {len(zoom_result)}")
                            logging.info(f"Types of elements in the list: {[type(item) for item in zoom_result]}")
                        next_round.append(zoom_result)
                        del zoom_result, zoom_matches
                        operation_id = f"operation_{operation_counter:03d}"
                        logging.info(f"  Logged operation {operation_id}") #: Frames {new_frame_range['indices']}")
                        operation_counter += 1


                else:
                
                    # Add stitched image to next round
                    if isinstance(stitched_image, list):
                        logging.info("stitched_imageis a list.")
                        logging.info(f"Number of elements in the list: {len(stitched_image)}")
                        logging.info(f"Types of elements in the list: {[type(item) for item in stitched_image]}")
                    next_round.append(stitched_image)
                    # new_frame_range = {
                    #     "indices": list(range(frames_list[i]["indices"][0], frames_list[i+1]["indices"][-1] + 1))
                    # }
                    # next_frames_list.append(new_frame_range)

                    # Log the stitching operation
                    operation_id = f"operation_{operation_counter:03d}"
                    # stitching_log = log_stitching(stitching_log, operation_id, frames_list[i], frames_list[i+1], new_frame_range, current_stitch_transform)
                    logging.info(f"  Logged operation {operation_id}") #: Frames {new_frame_range['indices']}")
                    operation_counter += 1  # Increment operation ID

                i += 2
            else:
                # Carry forward the last image
                logging.info(f"  Carrying forward the last image {i}") #: Frames {frames_list[i]['indices']}")
                if isinstance(image_list[i], list):
                    logging.info("image_list[i] is a list.")
                    logging.info(f"Number of elements in the list: {len(image_list[i])}")
                    logging.info(f"Types of elements in the list: {[type(item) for item in image_list[i]]}")
                next_round.append(image_list[i])
                # next_frames_list.append(frames_list[i])  # Carry forward its range
                i += 1

        # Memory cleanup after finishing this round
        del image_list, frames_list
        image_list = next_round
        frames_list = next_frames_list

        logging.info(f"  Next round images: {len(image_list)}")
        logging.info(f"  Next round frame ranges: {len(frames_list)}")

        iteration += 1

    logging.info(f"Final stitched image count: {len(image_list)}")
    logging.info(f"Final frame ranges count: {len(frames_list)}")
    return image_list, frames_list, stitching_log


def validate_images(image_list):
    for idx, image in enumerate(image_list):
        # Check if the element has the 'shape' attribute
        if not hasattr(image, 'shape'):
            logging.info(f"Invalid image at index {idx}: {type(image)}")
            logging.info(f"Content of invalid image: {image}")
        elif not isinstance(image, np.ndarray):
            logging.info(f"Image at index {idx} is not a valid NumPy array. Type: {type(image)}")
        else:
            logging.info(f"Valid image at index {idx} with shape: {image.shape}")


def stitch_images_pair_combos(image_list, disp): 
    crop_test = []
    if len(image_list) < 2:
        logging.info("Combos: Not enough images to stitch.")
        return image_list, crop_test, None

    validate_images(image_list)

    for i in range(len(image_list)):
        for j in range(len(image_list)):
            if i == j:
                continue

            # Find matches between images
            _, _, matches0, matches1, _, _ = find_best_rotation_matches(image_list[i], image_list[j], 20, disp)
            num_matches = len(matches0)

            if num_matches >= min_matches:
                stitched_image, num_matches, current_stitch_transform = stitch_two_images(image_list[i], image_list[j], disp)
                logging.info(f"Success! Stitched images {i} and {j}, with {num_matches} matches.")
                
                # Update image list
                new_image_list = [image_list[k] for k in range(len(image_list)) if k not in {i, j}] + [stitched_image]
                crop_test.append([i, j, num_matches, num_matches])

                # Cleanup
                del matches0, matches1, stitched_image
                gc.collect()
                torch.cuda.empty_cache()

                return new_image_list, crop_test, current_stitch_transform
            
            else:
                logging.info(f"Too few matches, {num_matches}, trying zoom.")
                zoom_result, zoom_matches, zoom_frames_list, zoom_stitching_log = zoom_and_stitch(
                    image_list[i], image_list[j], matches0, matches1, True
                )
                del matches0, matches1
                if zoom_matches < min_matches:
                    logging.info(f"Too few zoom matches, {zoom_matches}, trying split.")
                    del zoom_result, zoom_matches

                    split_result, split_matches, _, _ = split_and_stitch(image_list[i], image_list[j], True)
                    if split_matches < min_matches:
                        logging.info(f"  Too few split matches, {split_matches}, move  on.")
                        del split_result, split_matches
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        new_image_list = [image_list[k] for k in range(len(image_list)) if k not in {i, j}] + [split_result]
                        crop_test.append([i, j, num_matches, num_matches])
                        if isinstance(split_result, list):
                            logging.info("split_result is a list.")
                            logging.info(f"Number of elements in the list: {len(split_result)}")
                            logging.info(f"Types of elements in the list: {[type(item) for item in split_result]}")
                        del split_result, split_matches
                        gc.collect()
                        torch.cuda.empty_cache()
                        return new_image_list, crop_test, []
                else:
                    if isinstance(zoom_result, list):
                        logging.info("zoom_result is a list.")
                        logging.info(f"Number of elements in the list: {len(zoom_result)}")
                        logging.info(f"Types of elements in the list: {[type(item) for item in zoom_result]}")
                    new_image_list = [image_list[k] for k in range(len(image_list)) if k not in {i, j}] + [zoom_result]
                    crop_test.append([i, j, num_matches, num_matches])
                    del zoom_result, zoom_matches
                    gc.collect()
                    torch.cuda.empty_cache()
                    return new_image_list, crop_test, []

    logging.info("That's the best we can do, sorry!")
    return image_list, crop_test, None



def split_image(image):
    """
    Splits the given image in half either horizontally or vertically, depending on its longer side.
    
    Args:
        image (np.ndarray): The image in NumPy format or as a PyTorch tensor.
        
    Returns:
        tuple: Cropped halves of the image.
    """

    # Check if the image is on GPU
    is_cuda_tensor = isinstance(image, torch.Tensor) and image.is_cuda

    # Get image dimensions
    height, width = image.shape[:2]
    logging.info(f"Original image shape: {image.shape}")

    cropped_image1, cropped_image2 = None, None

    try:
        # Perform split based on the longer side
        if height >= width:  # Split horizontally
            cropped_image1 = image[: height // 2, :, ...]  # Top half
            cropped_image2 = image[height // 2:, :, ...]  # Bottom half
        else:  # Split vertically
            cropped_image1 = image[:, :width // 2, ...]  # Left half
            cropped_image2 = image[:, width // 2:, ...]  # Right half

        logging.info(f"Resulting shapes: {cropped_image1.shape}, {cropped_image2.shape}")

        # Move to CPU if needed
        if is_cuda_tensor:
            cropped_image1 = cropped_image1.cpu()
            cropped_image2 = cropped_image2.cpu()

    except Exception as e:
        logging.info(f"Error during image splitting: {e}")
        cropped_image1, cropped_image2 = None, None

    finally:
        # Explicit cleanup
        del height, width, image
        gc.collect()
        torch.cuda.empty_cache()

    return cropped_image1, cropped_image2


def get_bounding_box(matches0, matches1, margin=20, image_shape=None):
    """
    Given the matches, return the bounding box around the area with the dense matches.
    The margin is added around the bounding box to ensure enough context around the matches.

    Args:
        matches0: Tensor of shape (N, 2), containing the (x, y) coordinates of keypoints in the first image.
        matches1: Tensor of shape (N, 2), containing the (x, y) coordinates of keypoints in the second image.
        margin: Integer, the margin to apply around the bounding box.
        image_shape: Tuple (height, width), optional, to ensure bounding box stays within image bounds.

    Returns:
        bbox0: Bounding box (min_x, min_y, max_x, max_y) for the first image.
        bbox1: Bounding box (min_x, min_y, max_x, max_y) for the second image.
    """
    import torch

    # Ensure matches are tensors and remain on the same device
    matches0 = torch.as_tensor(matches0, device=matches0.device if isinstance(matches0, torch.Tensor) else "cpu")
    matches1 = torch.as_tensor(matches1, device=matches1.device if isinstance(matches1, torch.Tensor) else "cpu")

    # Compute min and max coordinates for matches0
    min_coords0 = torch.min(matches0, dim=0).values
    max_coords0 = torch.max(matches0, dim=0).values

    # Compute min and max coordinates for matches1
    min_coords1 = torch.min(matches1, dim=0).values
    max_coords1 = torch.max(matches1, dim=0).values

    # Free memory of matches tensors
    del matches0, matches1

    # Apply margin
    min_coords0 = torch.clamp(min_coords0 - margin, min=0)
    max_coords0 += margin
    min_coords1 = torch.clamp(min_coords1 - margin, min=0)
    max_coords1 += margin

    # Constrain bounding boxes to image dimensions if provided
    if image_shape:
        height, width = image_shape
        max_coords0[0] = min(max_coords0[0], width)
        max_coords0[1] = min(max_coords0[1], height)
        max_coords1[0] = min(max_coords1[0], width)
        max_coords1[1] = min(max_coords1[1], height)

    # Convert to tuples for output and free tensor memory
    bbox0 = (min_coords0[0].item(), min_coords0[1].item(), max_coords0[0].item(), max_coords0[1].item())
    bbox1 = (min_coords1[0].item(), min_coords1[1].item(), max_coords1[0].item(), max_coords1[1].item())

    del min_coords0, max_coords0, min_coords1, max_coords1

    return bbox0, bbox1


def zoom_and_stitch(image0_cv, image1_cv, matches0, matches1, disp, margin=20):
    logging.info("zoom and stitch")
    
    # Get the bounding boxes for both images based on matches with margin
    bbox0, bbox1 = get_bounding_box(matches0, matches1, margin)
    
    # Crop the images based on the bounding box to zoom in on the matching area
    cropped_image0 = image0_cv[int(bbox0[1]):int(bbox0[3]), int(bbox0[0]):int(bbox0[2])]
    cropped_image1 = image1_cv[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]

    logging.info(f"Image0 before zooming: {image0_cv.shape}, after zooming: {cropped_image0.shape}")
    logging.info(f"Image1 before zooming: {image1_cv.shape}, after zooming: {cropped_image1.shape}")

    # Debugging: Check if images are empty
    if cropped_image0.size == 0 or cropped_image1.size == 0:
        logging.info("Cropped image(s) are empty. Check bounding box coordinates.")
        return [image0_cv, image1_cv], 0, [], []

    # Recalculate the matches for the cropped region
    crop0, Rcrop1, cm0, cm1, angle, Rc = find_best_rotation_matches(cropped_image0, cropped_image1, 20, disp=True)
    num_matches = len(cm0)
    # Free cropped image memory if no sufficient matches
    if num_matches < min_matches:
        del cropped_image0, cropped_image1, crop0, Rcrop1, cm0, cm1, angle, Rc
        gc.collect()
        torch.cuda.empty_cache()  # Clear unused cached GPU memory
        logging.info(f"Zoom failed, only {num_matches} matches found.")
        return [image0_cv, image1_cv], num_matches, [], []

    # Attempt stitching the cropped images
    cropped_result, num_crop_matches, _ = stitch_two_images(cropped_image0, cropped_image1, True)

    # Free cropped memory after stitching
    del cropped_image0, cropped_image1, crop0, Rcrop1, angle, Rc
    gc.collect()
    torch.cuda.empty_cache()  # Clear unused cached GPU memory

    if num_crop_matches >= min_matches:
        logging.info(f"Success! Cropped images stitched with {num_crop_matches} matches. Proceeding with full stitching.")
        
        # Zoom stitching process
        zoom_result, zoom_frames_list, zoom_stitching_log = stitch_images_in_pairs(
            [cropped_result, image0_cv, image1_cv], [], [], True
        )
        
        # Free cropped result memory
        del cropped_result
        gc.collect()
        torch.cuda.empty_cache()  # Clear unused cached GPU memory
        if len(zoom_result) == 1: return zoom_result[0], len(cm0), zoom_frames_list, zoom_stitching_log
        

    logging.info(f"Zoom failed, only {num_crop_matches} matches found in cropped stitching.")
    return [image0_cv, image1_cv], len(cm0), [], []


def split_and_stitch(first, second, disp):
    logging.info("split and test")
    
    # Split images into halves
    cropped_first_1, cropped_first_2 = split_image(first)
    cropped_second_1, cropped_second_2 = split_image(second)

    # Perform stitching for the first set of halves
    result_1, num_matches_1, _ = stitch_two_images(cropped_first_1, cropped_second_1, disp)
    result_2, num_matches_2, _ = stitch_two_images(cropped_first_2, cropped_second_2, disp)

    if num_matches_1 >= min_matches:
        logging.info(f"Success! First halves of images stitched with {num_matches_1} matches. Proceed with full stitching.")
        
        # Stitch the results of the halves
        first_1, num_matches_11, _ = stitch_two_images(result_1, first, disp)
        second_1, num_matches_21, _ = stitch_two_images(result_1, second, disp)

        if num_matches_11 >= num_matches_21:
            print("first 1 with second")
            cropped_result, num_matches_cropped, _ = stitch_two_images(first_1, second, disp)
        else:
            print("second 1 with first")
            cropped_result, num_matches_cropped, _ = stitch_two_images(second_1, first, disp)

        # cropped_result, num_matches_cropped, _ = stitch_two_images(first_1, second_1, disp)
        # cropped_result, cropped_frames_list, cropped_stitching_log  = stitch_images_in_pairs([result_1, first, second], [], [], True)
        save_intermediate_images([result_1, first_1, second_1, cropped_result], output_folder, 11)
       
        del first_1, second_1, num_matches_11, num_matches_21
        gc.collect()
        torch.cuda.empty_cache()
        
        if num_matches_cropped >= min_matches:
            logging.info(f"Success! split and stitch worked!")
            del result_1, result_2, cropped_first_1, cropped_first_2, cropped_second_1, cropped_second_2
            gc.collect()
            torch.cuda.empty_cache()
            return cropped_result, num_matches_cropped, [], []
        else: 
            # logging.info(f"Fail, split and stitch didn't work. Left with {len(cropped_result)} images")
            logging.info(f"Fail, split and stitch didn't work. First {num_matches_11}, second {num_matches_21}, total {num_matches_cropped}, on to next strategy")
        

    if num_matches_2 >= min_matches:
        logging.info(f"Success! Second halves of images stitched with {num_matches_2} matches. Proceed with full stitching.")
        
        # Stitch the results of the halves
        first_2, num_matches_12, _ = stitch_two_images(result_2, first, disp)
        second_2, num_matches_22, _ = stitch_two_images(result_2, second, disp)

        if num_matches_12 >= num_matches_22:
            print("first 2 with second")
            cropped_result, num_matches_cropped, _ = stitch_two_images(first_2, second, disp)
        else:
            print("second 2 with first")
            cropped_result, num_matches_cropped, _ = stitch_two_images(second_2, first, disp)

        # cropped_result, num_matches_cropped, _ = stitch_two_images(first_2, second_2, disp)
        # cropped_result, cropped_frames_list, cropped_stitching_log  = stitch_images_in_pairs([result_2, first, second], [], [], True)
        save_intermediate_images([result_2, first_2, second_2, cropped_result], output_folder, 12)
        
        del first_2, second_2, num_matches_12, num_matches_22
        gc.collect()
        torch.cuda.empty_cache()

        if num_matches_cropped >= min_matches:
            logging.info(f"Success! split and stitch worked!")
            del result_1, result_2, cropped_first_1, cropped_first_2, cropped_second_1, cropped_second_2
            return cropped_result, num_matches_cropped, [], []
        else: 
            # logging.info(f"Fail, split and stitch didn't work. Left with {len(cropped_result)} images")
            logging.info(f"Fail, split and stitch didn't work. First {num_matches_12}, second {num_matches_22}, total {num_matches_cropped}, on to next strategy")
        

    del result_1, result_2
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"split failed, first w first matches: {num_matches_1}, second w second matches: {num_matches_2}.")
    logging.info(f"try again, first w second")

    # Reattempt stitching with swapped halves
    result_1, num_matches_1, _ = stitch_two_images(cropped_first_1, cropped_second_2, disp)
    result_2, num_matches_2, _ = stitch_two_images(cropped_first_2, cropped_second_1, disp)

    # cropped_first_1, cropped_first_2
    # cropped_second_1, cropped_second_2
    
    if num_matches_1 >= min_matches:
        logging.info(f"Success! First half of image0 with second half of image1 stitched with {num_matches_1} matches. Proceed with full stitching.")
        
        # Stitch the results of the halves
        first_1, num_matches_11, _ = stitch_two_images(result_1, first, disp)
        second_1, num_matches_21, _ = stitch_two_images(result_1, second, disp)

        if num_matches_11 >= num_matches_21:
            print("first 1 with second")
            cropped_result, num_matches_cropped, _ = stitch_two_images(first_1, second, disp)
        else:
            print("second 1 with first")
            cropped_result, num_matches_cropped, _ = stitch_two_images(second_1, first, disp)

        # cropped_result, num_matches_cropped, _ = stitch_two_images(first_1, second_1, disp)
        # cropped_result, cropped_frames_list, cropped_stitching_log  = stitch_images_in_pairs([result_1, first, second], [], [], True)
        save_intermediate_images([result_1, first_1, second_1, cropped_result], output_folder, 21)
        
        del first_1, second_1, num_matches_11, num_matches_21
        gc.collect()
        torch.cuda.empty_cache()
        
        if num_matches_cropped >= min_matches:
            logging.info(f"Success! split and stitch worked!")
            del result_1, result_2, cropped_first_1, cropped_first_2, cropped_second_1, cropped_second_2
            return cropped_result, num_matches_cropped, [], []
        else: 
            # logging.info(f"Fail, split and stitch didn't work. Left with {len(cropped_result)} images")
            logging.info(f"Fail, split and stitch didn't work. First {num_matches_11}, second {num_matches_21}, total {num_matches_cropped}, on to next strategy")
        

    if num_matches_2 >= min_matches:
        logging.info(f"Success! Second half of image0 with first half of image1 with {num_matches_2} matches. Proceed with full stitching.")
        
        # Stitch the results of the halves
        first_2, num_matches_12, _ = stitch_two_images(result_2, first, disp)
        second_2, num_matches_22, _ = stitch_two_images(result_2, second, disp)
        if num_matches_12 >= num_matches_22:
            print("first 2 with second")
            cropped_result, num_matches_cropped, _ = stitch_two_images(first_2, second, disp)
        else:
            print("second 2 with first")
            cropped_result, num_matches_cropped, _ = stitch_two_images(second_2, first, disp)

        # cropped_result, num_matches_cropped, _ = stitch_two_images(first_2, second_2, disp)
        # cropped_result, cropped_frames_list, cropped_stitching_log  = stitch_images_in_pairs([result_2, first, second], [], [], True)
        save_intermediate_images([result_2, first_2, second_2, cropped_result], output_folder, 22)

        del first_2, second_2
        gc.collect()
        torch.cuda.empty_cache()

        if num_matches_cropped >= min_matches:
            logging.info(f"Success! split and stitch worked!")
            del result_1, result_2, cropped_first_1, cropped_first_2, cropped_second_1, cropped_second_2
            return cropped_result, num_matches_cropped, [], []
        else: 
            # logging.info(f"Fail, split and stitch didn't work. Left with {len(cropped_result)} images")
            logging.info(f"Fail, split and stitch didn't work. First {num_matches_12}, second {num_matches_22}, total {num_matches_cropped}, on to next strategy")
        
    logging.info(f"split failed, first w second matches: {num_matches_1}, second w first matches: {num_matches_2}.")
    logging.info("no next strategy :( leaving split and stitch")

    del result_1, result_2, cropped_first_1, cropped_first_2, cropped_second_1, cropped_second_2

    # Return the original images if stitching fails
    return [first, second], num_matches_2, [], []

def crop_black_regions(image, min_crop_percent=0.1):
    """
    Crop out the black regions of the image while keeping a minimum percentage of the original image.
    Optimized for CUDA memory management using torch tensors.

    Args:
        image: Input image (NumPy array or PyTorch tensor).
        min_crop_percent: Minimum percentage of the original image to retain. Default is 10%.
    
    Returns:
        cropped_image: The image with black regions cropped.
    """

    # If the image is a NumPy array, convert it to a PyTorch tensor on the GPU
    if isinstance(image, np.ndarray):
        image = torch.tensor(image).cuda()  # Move image to GPU
    
    # Convert to grayscale using PyTorch operations
    gray_image = torch.mean(image.float(), dim=2, keepdim=True).byte() # Mean across the color channels
    gray_image = gray_image.squeeze()  # Remove the channel dimension
    
    # Apply a threshold to get a binary mask
    thresholded_image = gray_image > 1  # Non-black areas will have value > 0
    
    # Find the bounding box of non-black regions
    non_black_indices = torch.nonzero(thresholded_image)
    if non_black_indices.size(0) > 0:
        min_x = non_black_indices[:, 1].min().item()
        max_x = non_black_indices[:, 1].max().item()
        min_y = non_black_indices[:, 0].min().item()
        max_y = non_black_indices[:, 0].max().item()

        # Crop the image based on the bounding box
        cropped_image = image[min_y:max_y, min_x:max_x]
        
        # Calculate the area of the cropped image and the original image
        cropped_area = cropped_image.numel()
        original_area = image.numel()

        # Fail-safe: Check if cropped area is less than min_crop_percent of original area
        if cropped_area / original_area < min_crop_percent:
            logging.info("Cropped area is too small, returning the original image.")
            return image.cpu().numpy()  # Return original image as NumPy array
    else:
        # If no non-black pixels, return original image
        cropped_image = image
    
    # Cleanup CUDA memory
    del gray_image, thresholded_image, non_black_indices
    gc.collect()
    torch.cuda.empty_cache()

    # Return the cropped image as a NumPy array
    return cropped_image.cpu().numpy()

