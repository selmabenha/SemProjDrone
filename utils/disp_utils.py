# from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet
# from lightglue.utils import load_image, rbd
from lightglue import viz2d
import logging
import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np
import logging

# logging.basicConfig(
#     level=logging.DEBUG,  # Set the minimum level of messages to capture
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("logs/script.log"),  # Write logs to this file
#         logging.StreamHandler()  # Optionally, also logging.info to console
#     ]
# )

def display_extracted_matches(matches, image0, image1, m_kpts0, m_kpts1, kpts0, kpts1, matches01, output_folder):
    length_matches = len(m_kpts0)
    print(f"len(matches) = {length_matches}")
    image0_np = image0.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    image1_np = image1.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    
    # Ensure that the images are in the correct range (0-255) and are of type uint8
    image0_np = np.clip(image0_np * 255, 0, 255).astype(np.uint8)
    image1_np = np.clip(image1_np * 255, 0, 255).astype(np.uint8)

    # Step 1: Create blank canvas large enough for both images side by side with padding
    height0, width0 = image0_np.shape[:2]
    height1, width1 = image1_np.shape[:2]

    # Add padding (gap) between the images
    padding = 50  # You can adjust this value for the desired gap
    canvas_height = max(height0, height1)
    canvas_width = width0 + width1 + padding  # Add padding to the width

    # Create a new canvas with combined width, maximum height, and padding
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place image0 on the left and image1 on the right of the canvas with padding between
    canvas[:height0, :width0] = image0_np
    canvas[:height1, width0 + padding:width0 + padding + width1] = image1_np

    # Step 2: Draw the matches between keypoints (connecting lines between m_kpts0 and m_kpts1)
    for i in range(length_matches):
        pt0 = tuple(map(int, m_kpts0[i]))  # keypoint in image0
        pt1 = tuple(map(int, m_kpts1[i]))  # corresponding keypoint in image1
        # Draw a line between the keypoints
        cv2.line(canvas, pt0, (pt1[0] + width0 + padding, pt1[1]), (0, 255, 0), 1)
        # Draw a dot (circle) at pt0
        cv2.circle(canvas, pt0, 3, (0, 255, 0), -1)  # Red dot, filled
        # Draw a dot (circle) at pt1
        cv2.circle(canvas, (pt1[0] + width0 + padding, pt1[1]), 3, (0, 255, 0), -1)  # Blue dot, filled

    # Step 3: Save the image with matches to disk
    match_image_filename = os.path.join(output_folder, f"matched_keypoints_image_{length_matches}.png")
    if not cv2.imwrite(match_image_filename, canvas):
        print(f"Error writing extracted features image {match_image_filename}")
        try:
            img = Image.fromarray(canvas)
            img.save(match_image_filename)
            print(f"Image saved successfully to {match_image_filename}")
        except Exception as e:
            print(f"Error saving image {match_image_filename}: {e}")
    else:
        print(f"Saved extracted features Image to {match_image_filename}")

    # Step 4: Create another canvas for pruned keypoints
    canvas_pruned = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_pruned[:height0, :width0] = image0_np
    canvas_pruned[:height1, width0 + padding:width0 + padding + width1] = image1_np

    # Step 5: Plot the pruned keypoints on the new canvas
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    for kp0, kp1 in zip(kpts0, kpts1):
        # Draw pruned keypoints (just points)
        cv2.circle(canvas_pruned, tuple(map(int, kp0)), 5, (255, 0, 0), -1)
        cv2.circle(canvas_pruned, tuple(map(int, (kp1[0] + width0 + padding, kp1[1]))), 5, (0, 0, 255), -1)

    # Step 6: Save the image with pruned keypoints to disk
    pruned_image_filename = os.path.join(output_folder, f"pruned_keypoints_image_{length_matches}.png")
    if not cv2.imwrite(pruned_image_filename, canvas_pruned):
        print(f"Error writing pruned features image {pruned_image_filename}")
        try:
            img = Image.fromarray(canvas_pruned)
            img.save(pruned_image_filename)
            print(f"Image saved successfully to {pruned_image_filename}")
        except Exception as e:
            print(f"Error saving image {pruned_image_filename}: {e}")
    else:
        print(f"Saved pruned features Image to {pruned_image_filename}")


def display_transformed_images(image0_transformed, image1_transformed, length_matches, output_folder):
    # Define the padding (gap) between the images
    padding = 50  # You can adjust this value for the desired gap
    
    # Get the dimensions of the transformed images
    height0, width0 = image0_transformed.shape[:2]
    height1, width1 = image1_transformed.shape[:2]
    
    # Create a canvas large enough to hold both images side by side with padding in between
    transformed_width = width0 + width1 + padding  # Add padding between the images
    transformed_height = max(height0, height1)  # Take the larger height
    transformed_canvas = np.zeros((transformed_height, transformed_width, 3), dtype=np.uint8)

    # Place the first image on the canvas
    transformed_canvas[:height0, :width0] = image0_transformed
    # Place the second image on the canvas with padding
    transformed_canvas[:height1, width0 + padding:width0 + padding + width1] = image1_transformed

    # Save the transformed canvas (side by side images) to a file
    transformed_filename = os.path.join(output_folder, f"transformed_images_{length_matches}.jpg")
    if not cv2.imwrite(transformed_filename, transformed_canvas):
        print(f"Error writing image {transformed_filename}")
        try:
            img = Image.fromarray(transformed_canvas)
            img.save(transformed_filename)
            print(f"Image saved successfully to {transformed_filename}")
        except Exception as e:
            print(f"Error saving image {transformed_filename}: {e}")
    else:
        print(f"Saved transformed images to {transformed_filename}")


def display_original_merged(first, second, merged, i, output_folder):
    # Add padding (gap) between images
    padding = 50  # You can adjust this value for the desired gap
    
    # Get the largest image dimensions
    height, width, _ = merged.shape

    # Create a blank canvas to hold all three images side by side with padding
    combined_width = first.shape[1] + second.shape[1] + merged.shape[1] + 2 * padding  # Account for padding
    combined_height = max(first.shape[0], second.shape[0], merged.shape[0])
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Place each image on the combined canvas with padding
    combined_image[:first.shape[0], :first.shape[1]] = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    combined_image[:second.shape[0], first.shape[1] + padding:first.shape[1] + padding + second.shape[1]] = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
    combined_image[:merged.shape[0], first.shape[1] + second.shape[1] + 2 * padding:] = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

    # Add titles for each image (manually as part of the image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 1
    title_thickness = 2
    title_color = (255, 255, 255)  # White color for title text

    # Title positions with padding considered
    title0_position = (int(first.shape[1] / 2) - 50, 30)
    title1_position = (int(first.shape[1] + padding + second.shape[1] / 2) - 50, 30)
    title2_position = (int(first.shape[1] + padding + second.shape[1] + padding + merged.shape[1] / 2) - 50, 30)

    # Add titles
    cv2.putText(combined_image, f'Image {i}', title0_position, font, title_font_scale, title_color, title_thickness)
    cv2.putText(combined_image, f'Image {i+1}', title1_position, font, title_font_scale, title_color, title_thickness)
    cv2.putText(combined_image, 'Stitched', title2_position, font, title_font_scale, title_color, title_thickness)

    # Save the combined image
    combined_image_filename = os.path.join(output_folder, f"combined_images_{i}.jpg")
    if not cv2.imwrite(combined_image_filename, combined_image):
        print(f"Error writing image {combined_image_filename}")
        try:
            img = Image.fromarray(combined_image)
            img.save(combined_image_filename)
            print(f"Image saved successfully to {combined_image_filename}")
        except Exception as e:
            print(f"Error saving image {combined_image_filename}: {e}")
    else:
        print(f"Saved combined images to {combined_image_filename}")
