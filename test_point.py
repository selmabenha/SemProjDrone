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
from utils import *

from PIL import Image


logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/log_stitching.json"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

print("Script started - test")

torch.set_grad_enabled(False)
images = Path("/home/finette/VideoStitching/selma/test/test_imgs")
extract_frames_done = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = DoGHardNet(max_num_keypoints=None).eval().to(device)  # load the extractor
matcher = LightGlue(features="doghardnet").eval().to(device)
print("extractor and matcher done")

frame_step = 50
overlap_threshold = 2000  # Threshold for sufficient overlap
min_matches = 1000  # Threshold for minimal matches before there is a problem

output_images = "/home/finette/VideoStitching/selma/test/output_imgs"
if not os.path.exists(output_images):
    os.makedirs(output_images)

extract_frames_done = True


image_paths = sorted(list(images.glob("*.jpg")))  # Adjust this to the path of your images

images_cv, frame_ranges = get_images_frames(image_paths)
print(f"len(images_cv) = {len(images_cv)}, len(frame_ranges) = {len(frame_ranges)}")
print(f"frame_ranges = {frame_ranges}")

A = (100, 200)
B = (200, 100)
C = (300, 300)
D = (500, 800)
E = (1000, 200)

cv2.circle(images_cv[0], A, 10, (0, 0, 255), -1)
cv2.circle(images_cv[1], B, 10, (0, 0, 255), -1)
cv2.circle(images_cv[2], C, 10, (0, 0, 255), -1)
cv2.circle(images_cv[3], D, 10, (0, 0, 255), -1)
cv2.circle(images_cv[4], E, 10, (0, 0, 255), -1)

# Get Transformed Points
# result_cv, all_transform_matrices, full_frames_list = stitch_images_in_pairs(images_cv, frame_ranges, True)
# print(f"# of images is {len(result_cv)}")
# print(f"all transform matrices {all_transform_matrices}")
# print(f"full_frames_list {full_frames_list}")

result_cv = cv2.imread("/home/finette/VideoStitching/selma/test/output_imgs/result_test.jpg")


all_transform_matrices = [[
    [[[1.0028871297836304, 0.006782423239201307, 38.73857498168945], 
    [0.00584684731438756, 1.0007307529449463, 610.93701171875], 
    [3.56135205947794e-06, 1.161503041657852e-05, 1.0]], 
    [[1.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], 
    [-0.0, 1.0, 0.0]], 
    [[np.float64(1.015625), np.float64(1.5462962962962963)], 
    [np.float64(1.015625), np.float64(1.5462962962962963)]]], 
    
    [[[1.0050134658813477, -0.014084731228649616, 30.038944244384766], 
    [0.01434276346117258, 1.0135592222213745, 604.8275756835938], 
    [3.499680815366446e-06, -1.106988565879874e-05, 1.0]], 
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], [-0.0, 1.0, 0.0]], 
    [[np.float64(1.0177083333333334), np.float64(1.6074074074074074)], 
    [np.float64(1.0177083333333334), np.float64(1.6074074074074074)]]]
    ], [
    
    [[[1.0481579303741455, -0.025564925745129585, 37.67458724975586], 
    [0.01847204566001892, 0.9809253215789795, 1273.6826171875], 
    [1.5411851563840173e-05, -1.799422716430854e-05, 1.0]], 
    [[1.0, 0.0, 5.0], 
    [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], 
    [-0.0, 1.0, 0.0]], 
    [[np.float64(1.0476923076923077), np.float64(1.7976047904191617)], 
    [np.float64(1.0455475946775845), np.float64(1.7292626728110598)]]]
    ], [
    
    [[[0.9853286743164062, -0.00844452902674675, 39.86235809326172], 
    [-0.022431207820773125, 1.1037862300872803, 564.7452392578125], 
    [-7.104961241566343e-06, -1.4271906366047915e-05, 1.0]], 
    [[1.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], 
    [-0.0, 1.0, 0.0]], 
    [[np.float64(1.0523739598629467), np.float64(1.3540972684876749)], 
    [np.float64(1.1197916666666667), np.float64(3.763888888888889)]]]
    ]]
full_frames_list = [[(1, 299), (300, 549), (550, 799), (800, 1049), (1050, 7516)], [(1, 549), (550, 1049), (1050, 7516)], [(1, 1049), (1050, 7516)], [(1, 7516)]]


# cv2.imwrite(f'{output_images}/result_test.jpg', result_cv[0])
# cv2.imwrite(f'{output_images}/image_original_0.jpg', images_cv[0])
# cv2.imwrite(f'{output_images}/image_original_1.jpg', images_cv[1])
# cv2.imwrite(f'{output_images}/image_original_2.jpg', images_cv[2])
# cv2.imwrite(f'{output_images}/image_original_3.jpg', images_cv[3])
# cv2.imwrite(f'{output_images}/image_original_4.jpg', images_cv[4])

frame_a = {"indices": list(range(1, 300))}
frame_b = {"indices": list(range(300, 549))}
frame_c = {"indices": list(range(550, 799))}
frame_d = {"indices": list(range(800, 1049))}
frame_e = {"indices": list(range(1050, 1299))}

stitching_log = []

def log_stitching(operation_id, frame_a, frame_b, result_frame, transformation_matrices):
    """
    Log the stitching operation with transformation matrices and frame indices.

    :param operation_id: Identifier for the stitching operation
    :param frame_a: First frame (with indices)
    :param frame_b: Second frame (with indices)
    :param result_frame: Resulting frame (with indices)
    :param transformation_matrices: List of transformation matrices used in this operation
    """
    entry = {
        "operation_id": operation_id,
        "input_frames": [
            {"frame_indices": frame_a["indices"]},
            {"frame_indices": frame_b["indices"]}
        ],
        "result_frame": {
            "frame_indices": result_frame["indices"]
        },
        "transformation_matrices": transformation_matrices  # Store as strings for clarity
    }
    stitching_log.append(entry)

operation_id = 0

# Stitching frame_a and frame_b
result_frame_ab = {
    "indices": frame_a["indices"] + frame_b["indices"]
}

[H0, T0, R0, S0] = [
    [[1.0028871297836304, 0.006782423239201307, 38.73857498168945], 
    [0.00584684731438756, 1.0007307529449463, 610.93701171875], 
    [3.56135205947794e-06, 1.161503041657852e-05, 1.0]], 
    [[1.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], 
    [-0.0, 1.0, 0.0]], 
    [[np.float64(1.015625), np.float64(1.5462962962962963)], 
    [np.float64(1.015625), np.float64(1.5462962962962963)]]
]

# Log the first stitch
log_stitching(operation_id, frame_a, frame_b, result_frame_ab, [H0, T0, R0, S0])

# For the second operation, stitch result_frame_ab with frame_c
operation_id = 1

result_frame_cd = {
    "indices": frame_c["indices"] + frame_d["indices"]
}

[H1, T1, R1, S1] = [
    [[1.0050134658813477, -0.014084731228649616, 30.038944244384766], 
    [0.01434276346117258, 1.0135592222213745, 604.8275756835938], 
    [3.499680815366446e-06, -1.106988565879874e-05, 1.0]], 
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], [-0.0, 1.0, 0.0]], 
    [[np.float64(1.0177083333333334), np.float64(1.6074074074074074)], 
    [np.float64(1.0177083333333334), np.float64(1.6074074074074074)]]
]

# Log the second stitch
log_stitching(operation_id, frame_c, frame_d, result_frame_cd, [H1, T1, R1, S1])

# For the third, ab with cd
operation_id = 2

result_frame_abcd = {
    "indices": result_frame_ab["indices"] + result_frame_cd["indices"]
}

[H2, T2, R2, S2] = [
    [[1.0481579303741455, -0.025564925745129585, 37.67458724975586], 
    [0.01847204566001892, 0.9809253215789795, 1273.6826171875], 
    [1.5411851563840173e-05, -1.799422716430854e-05, 1.0]], 
    [[1.0, 0.0, 5.0], 
    [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], 
    [-0.0, 1.0, 0.0]], 
    [[np.float64(1.0476923076923077), np.float64(1.7976047904191617)], 
    [np.float64(1.0455475946775845), np.float64(1.7292626728110598)]]
]

# Log the third stitch
log_stitching(operation_id, result_frame_ab, result_frame_cd, result_frame_abcd, [H2, T2, R2, S2])

# For the fourth, abcd with e
operation_id = 3

result_frame_abcde = {
    "indices": result_frame_abcd["indices"] + frame_e["indices"]
}

[H3, T3, R3, S3] = [
    [[0.9853286743164062, -0.00844452902674675, 39.86235809326172], 
    [-0.022431207820773125, 1.1037862300872803, 564.7452392578125], 
    [-7.104961241566343e-06, -1.4271906366047915e-05, 1.0]], 
    [[1.0, 0.0, 0.0], 
    [0.0, 1.0, 0.0]], 
    [[1.0, 0.0, 0.0], 
    [-0.0, 1.0, 0.0]], 
    [[np.float64(1.0523739598629467), np.float64(1.3540972684876749)], 
    [np.float64(1.1197916666666667), np.float64(3.763888888888889)]]
]

# Log the fourth stitch
log_stitching(operation_id, result_frame_abcd, frame_e, result_frame_abcde, [H3, T3, R3, S3])



# Initialize A_new, B_new, and C_new
A_new, B_new, C_new, D_new, E_new = A, B, C, D, E

for entry in stitching_log:
    operation_id = entry["operation_id"]
    input_frames = entry["input_frames"]
    result_frame = entry["result_frame"]
    [H, T, R, S] = entry["transformation_matrices"]
    print(f"operation # {operation_id}")

    # Extract the frame indices from the input frames
    first_frame_indices = input_frames[0]["frame_indices"]
    second_frame_indices = input_frames[1]["frame_indices"]
    result_frame_indices = result_frame["frame_indices"]
    print(f"A_prev = {A_new}, B_prev = {B_new}, C_prev = {C_new}, D_prev = {D_new}, E_prev = {E_new}")

    # Prepare points for transformation
    first = []
    second = []
    point_map = {}  # Map original points to their transformed counterparts

    # Check and append A
    if set(frame_a["indices"]).issubset(set(first_frame_indices)):
        print("A first")
        first.append(A_new)
        point_map["A"] = "first"
    elif set(frame_a["indices"]).issubset(set(second_frame_indices)):
        print("A second")
        second.append(A_new)
        point_map["A"] = "second"

    # Check and append B
    if set(frame_b["indices"]).issubset(set(first_frame_indices)):
        print("B first")
        first.append(B_new)
        point_map["B"] = "first"
    elif set(frame_b["indices"]).issubset(set(second_frame_indices)):
        print("B second")
        second.append(B_new)
        point_map["B"] = "second"

    # Check and append C
    if set(frame_c["indices"]).issubset(set(first_frame_indices)):
        print("C first")
        first.append(C_new)
        point_map["C"] = "first"
    elif set(frame_c["indices"]).issubset(set(second_frame_indices)):
        print("C second")
        second.append(C_new)
        point_map["C"] = "second"

    # Check and append D
    if set(frame_d["indices"]).issubset(set(first_frame_indices)):
        print("D first")
        first.append(D_new)
        point_map["D"] = "first"
    elif set(frame_d["indices"]).issubset(set(second_frame_indices)):
        print("D second")
        second.append(D_new)
        point_map["D"] = "second"

    # Check and append E
    if set(frame_e["indices"]).issubset(set(first_frame_indices)):
        print("E first")
        first.append(E_new)
        point_map["E"] = "first"
    elif set(frame_e["indices"]).issubset(set(second_frame_indices)):
        print("E second")
        second.append(E_new)
        point_map["E"] = "second"

    # Transform points
    first_new, second_new = transform_points(first, second, H, T, R, S)

    # Map transformed points back to variables
    for i, point in enumerate(first):
        if np.array_equal(point, A_new):
            A_new = first_new[i]
        elif np.array_equal(point, B_new):
            B_new = first_new[i]
        elif np.array_equal(point, C_new):
            C_new = first_new[i]
        elif np.array_equal(point, D_new):
            D_new = first_new[i]
        elif np.array_equal(point, E_new):
            E_new = first_new[i]

    for i, point in enumerate(second):
        if np.array_equal(point, A_new):
            A_new = second_new[i]
        elif np.array_equal(point, B_new):
            B_new = second_new[i]
        elif np.array_equal(point, C_new):
            C_new = second_new[i]
        elif np.array_equal(point, D_new):
            D_new = second_new[i]
        elif np.array_equal(point, E_new):
            E_new = second_new[i]

    print(f"A_new = {A_new}, B_new = {B_new}, C_new = {C_new}, D_new = {D_new}, E_new = {E_new}")

        
# A_new, B_new = transform_first_point_and_B_with_rotation(A, B, H, T, R)
logging.info(f"A = {A}, A_new = {A_new}, B = {B}, B_new = {B_new}, C = {C}, C_new = {C_new}, D = {D}, D_new = {D_new}, E = {E}, E_new = {E_new}")

# Convert the points to integers before passing them to cv2.circle
A_new_int = (int(A_new[0]), int(A_new[1]))
B_new_int = (int(B_new[0]), int(B_new[1]))
C_new_int = (int(C_new[0]), int(C_new[1]))
D_new_int = (int(D_new[0]), int(D_new[1]))
E_new_int = (int(E_new[0]), int(E_new[1]))

# Now, use these integer points in the circle function
cv2.circle(result_cv, A_new_int, 15, (255, 0, 0), -1)
cv2.circle(result_cv, B_new_int, 15, (255, 0, 0), -1)
cv2.circle(result_cv, C_new_int, 15, (255, 0, 0), -1)
cv2.circle(result_cv, D_new_int, 15, (255, 0, 0), -1)
cv2.circle(result_cv, E_new_int, 15, (255, 0, 0), -1)


cv2.imwrite(f'{output_images}/result_ABC.jpg', result_cv)

# display_original_merged(images_cv[0], images_cv[1], result_cv[0], 1, output_images)
