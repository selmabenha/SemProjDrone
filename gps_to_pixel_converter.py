from pathlib import Path

# from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet
# import torch
import numpy as np
import cv2
import re
import csv
from utils import *
# from scipy import stats
import math

# Haversine formula for distance in meters
def haversine(gps_coords1, gps_coords2):
    lat1, lon1 = gps_coords1
    lat2, lon2 = gps_coords2
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

# # GPS coordinates for each frame
# frames = {
#     "Frame 0001": {
#         "Center": (38.012993, 23.753021),
#         "BL": (38.01294866671684, 23.751500607975878),
#         "BR": (38.01195110782173, 23.7539455905554),
#         "TL": (38.014075006024996, 23.752020644025926),
#         "TR": (38.01313321508272, 23.754519839640412),
#     },
#     "Frame 550": {
#         "Center": (38.014368, 23.753697),
#         "BL": (38.01429970916754, 23.752094765825955),
#         "BR": (38.01337169233648, 23.754610394677393),
#         "TL": (38.015444105584216, 23.752723678801352),
#         "TR": (38.01451567082068, 23.755209580243115),
#     },
#     "Frame 1550": {
#         "Center": (38.017046, 23.755012),
#         "BL": (38.016958911958724, 23.753283414549035),
#         "BR": (38.016000059967155, 23.7559713929887),
#         "TL": (38.01817969234928, 23.75391872957148),
#         "TR": (38.01717569642845, 23.756610446401968),
#     },
#     "Frame 2050": {
#         "Center": (38.018388, 23.755672),
#         "BL": (38.0182631116391, 23.753898651103516),
#         "BR": (38.01739876566146, 23.756559505254675),
#         "TL": (38.01960176594627, 23.75452214844139),
#         "TR": (38.018768659754635, 23.75664191768973),
#     },
# }

# # Calculate distances for each frame
# results = {}
# for frame, coords in frames.items():
#     center = coords["Center"]
#     BL = coords["BL"]
#     BR = coords["BR"]
#     TL = coords["TL"]
#     TR = coords["TR"]
    
#     # Side lengths
#     widthB = haversine(BL, BR)
#     widthT = haversine(TL, TR)
#     heightL = haversine(BL, TL)
#     heightR = haversine(BR, TR)
    
#     # Distances to center
#     BL_to_center = haversine(center, BL)
#     BR_to_center = haversine(center, BR)
#     TL_to_center = haversine(center, TL)
#     TR_to_center = haversine(center, TR)
    
#     results[frame] = {
#         "Width - Bottom": widthB,
#         "Width - Top": widthT,
#         "Height - Left": heightL,
#         "Height - Right": heightR,
#         "BL-Center": BL_to_center,
#         "BR-Center": BR_to_center,
#         "TL-Center": TL_to_center,
#         "TR-Center": TR_to_center,
#     }

# all_widths, all_heights, all_centers = [], [], []
# def get_avg_distances(results):
#     for frame, data in results.items():
#         all_widths.extend([data['Width - Top'], data['Width - Bottom']])
#         all_heights.extend([data['Height - Left'], data['Height - Right']])
#         all_centers.extend([data['BL-Center'], data['BR-Center'], data['TL-Center'], data['TR-Center']])

# get_avg_distances(results)
# FoV_vertical = np.round(np.mean(all_heights))
# print(f"width mean = {np.mean(all_widths)}, median = {np.median(all_widths)}, with std = {np.std(all_widths, ddof=1)}")
# print(f"height mean = {np.mean(all_heights)}, median = {np.median(all_heights)}, with std = {np.std(all_heights, ddof=1)}")
# print(f"center mean = {np.mean(all_centers)}, median = {np.median(all_centers)}, with std = {np.std(all_centers, ddof=1)}")

# input_gps_file = "/Users/selmabenhassine/Desktop/SemProjDrone/DJI_0763.SRT"
# images_folder = Path("/Users/selmabenhassine/Desktop/SemProjDrone/extracted_frames")
# extracted_frames = sorted(list(images_folder.glob("*.jpg")))  # Adjust this to the path of your images

reference_frames = [1, 300, 550, 800, 1050, 1300, 1550, 1800, 2050, 2350, 2600, 2900, 3200, 3500, 3800, 4100, 4350, 4600, 4900, 5150, 5400, 5650, 6050, 6300, 6550, 6800, 7517]

# def extract_coords(coord):
#     regex = r"\[latitude\s*:\s*([-\d.]+)\]\s*\[longtitude\s*:\s*([-\d.]+)\]"
#     match = re.search(regex, coord)
#     if match:
#         latitude, longitude = float(match.group(1)), float(match.group(2))
#         return (latitude, longitude)

# # with open(input_gps_file, 'r') as file:
# #     drone_coords_file = file.read()
# #     raw_coords = [c for c in drone_coords_file.split('\n\n') if c]
# #     drone_coords = [extract_coords(r) for r in raw_coords]

# def calculate_displacement_to_ref(index):
#     global index_ref
#     if index == 0:
#         return 0.0
#     elif index == reference_frames[index_ref+1]:
#         index_ref += 1
#         return 0.0
#     else:
#         gps_curr = drone_coords[index]
#         gps_prev = drone_coords[reference_frames[index_ref] - 1]
#         return (haversine(gps_curr, gps_prev))

######
# # images_cv, _ = get_images_frames(extracted_frames)
# # images_py = [convert_cv_to_py(image) for image in images_cv]

tracking_files = list(Path("/Users/selmabenhassine/Desktop/SemProjDrone/tracking_small").glob("*.txt"))
index_ref = 0  # Initialize index_ref

def frame_pix_displacement(index):
    global index_ref
    if index == 0:
        return [0.0]
    elif index == reference_frames[index_ref+1] and index != 1:
        index_ref += 1
        return [0.0]
    else:
        # Load the points for the current and reference frames
        points_ref = load_track_points(Path(tracking_files[index_ref]))
        points = load_track_points(Path(tracking_files[index]))

        # Extract only the y-coordinates for points with matching class_id
        y_displacements = []
        for point, point_ref in zip(points, points_ref):
            coords, metadata = point
            coords_ref, metadata_ref = point_ref

            # Extract class_ids
            class_id = metadata[1].strip()  # Assuming the class_id is stored as a string
            class_id_ref = metadata_ref[1].strip()

            # If class_ids match, compute y-displacement
            if class_id == class_id_ref:
                y_disp = coords[:, 1] - coords_ref[:, 1]
                y_disp_mean = np.mean(y_disp)
                y_displacements.append(y_disp_mean)
            else:
                y_displacements.append(None)

        return y_displacements

# Iterate over the tracking files
all_y_pix_displacement = [frame_pix_displacement(index) 
                          for index, _ in enumerate(tracking_files)]

all_y_pix_median = [np.median(
                    [val for val in frame_disp if val is not None]) 
                    for frame_disp in all_y_pix_displacement]

# Replace None values in all_y_pix_displacement with the corresponding median
all_y_pix_displacement_median = [
    [
        val if val is not None else all_y_pix_median[frame_index]
        for val in frame_disp
    ]
    for frame_index, frame_disp in enumerate(all_y_pix_displacement)
]

for frame_disp in all_y_pix_displacement: print(frame_disp)
# # # Use list comprehension to get median displacements for all consecutive image pairs

# # Get only displacement vectors of reference frames
# all_gps_displacement = [calculate_displacement_to_ref(index) for index, _ in enumerate(drone_coords)]
# ref_gps_displacement = [all_gps_displacement[frame - 1] for frame in reference_frames]

# # # Calculate the scale (pixel-to-GPS conversion factor) for each pair of displacements
# # gps_to_pix_scale = [
# #     ((pixel[0].item() / gps[0]) if gps[0] else 0, 
# #     (pixel[1].item() / gps[1]) if gps[1] else 0) 
# #     for gps, pixel in zip(ref_gps_displacement_vectors, ref_pixel_displacement_vectors_medians)
# # ]

# # images_cv, _ = get_images_frames(extracted_frames)

# def get_image_heights(image_paths):
#     heights = []
#     for image_path in image_paths:
#         image = cv2.imread(str(image_path))  # Read the image
#         if image is not None:
#             height, width, _ = image.shape  # Get dimensions (height, width, channels)
#             if (len(heights) > 1 and height != heights[-1]) or len(heights) == 0:
#                 heights.append(height)
#         else:
#             print(f"Warning: Could not read {image_path}")
#             heights.append(None)  # Append None for unreadable images
#     return heights

# image_heights = get_image_heights(extracted_frames)
# print(f"image heights are : {image_heights}")

# gps_to_pix_scale = image_heights[0]/FoV_vertical
# print(f"gps to pix scale is : {gps_to_pix_scale}")

# all_pixel_displacement = np.array(all_gps_displacement) * gps_to_pix_scale
# ref_pixel_displacement = np.array(ref_gps_displacement) * gps_to_pix_scale

# with open('drone_coords_new.csv', 'w', newline='') as file:
#     columns = [
#         'frame_number',
#         'latitude',
#         'longitude',
#         'gps_displacement_wrt_ref',
#         'pix_displacement_wrt_ref',
#         'is_reference'
#         ]
#     data = [[
#         index + 1,
#         coords[0],
#         coords[1],
#         all_gps_displacement[index],
#         all_pixel_displacement[index],
#         index + 1 in reference_frames]
#         for index, coords in enumerate(drone_coords)]

#     writer = csv.writer(file)
#     writer.writerow(columns)
#     writer.writerows(data)



# # # GPS coordinates
# # center0 = (38.014368, 23.753697)  # Center point (latitude, longitude)
# # bottomL0 = (38.01429970916754, 23.752094765825955)     # Top point
# # bottomR0 = (38.01337169233648, 23.754610394677393)     # Top point

# # center1 = (38.017046, 23.755012)  # Center point (latitude, longitude)
# # bottomL1 = (38.016958911958724, 23.753283414549035)     # Top point
# # bottomR1 = (38.016000059967155, 23.7559713929887)     # Top point

# # # Calculate width and height
# # width0 = haversine(bottomL0, bottomR0)
# # width1 = haversine(bottomL1, bottomR1)

# # print(f"Width between bottomL0 and bottomR0: {width0} m")
# # print(f"Width between bottomL1 and bottomR1: {width1} m")

# # checkL0 = haversine(center0, bottomL0)
# # checkR0 = haversine(center0, bottomR0)

# # print(f"Distance from center0 to bottomL0: {checkL0} m")
# # print(f"Distance from center0 to bottomR0: {checkR0} m")

# # checkL1 = haversine(center1, bottomL1)
# # checkR1 = haversine(center1, bottomR1)

# # print(f"Distance from center1 to bottomL1: {checkL1} m")
# # print(f"Distance from center1 to bottomR1: {checkR1} m")

# # # Given values
# # focal_length = 280  # in mm
# # fov_width = 250  # in meters
# # altitude = 401.431  # in meters

# # # Step 1: Calculate the FoV angle (θ)
# # theta = math.atan(fov_width / (2 * altitude))

# # # Step 2: Calculate the sensor size (S) in mm
# # sensor_size = 2 * focal_length * math.tan(theta)

# # print(f"Estimated sensor size: {sensor_size:.2f} mm")


# def calculate_fov(sensor_width, sensor_height, focal_length, working_distance):
#     # Calculate FoV in radians
#     fov_horizontal = 2 * math.atan(sensor_width / (2 * focal_length))
#     fov_vertical = 2 * math.atan(sensor_height / (2 * focal_length))
    
#     # Convert FoV to degrees
#     fov_horizontal_deg = math.degrees(fov_horizontal)
#     fov_vertical_deg = math.degrees(fov_vertical)
    
#     # Calculate dimensions at the working distance
#     horizontal_fov_at_distance = 2 * working_distance * math.tan(fov_horizontal / 2)
#     vertical_fov_at_distance = 2 * working_distance * math.tan(fov_vertical / 2)
    
#     return fov_horizontal_deg, fov_vertical_deg, horizontal_fov_at_distance, vertical_fov_at_distance


# def calculate_projected_image_size(sensor_width, sensor_height, focal_length, working_distance):
#     """
#     Calculate the projected image size in the real world based on sensor dimensions,
#     focal length, and working distance.

#     Parameters:
#     - sensor_width (float): Width of the camera sensor in mm.
#     - sensor_height (float): Height of the camera sensor in mm.
#     - focal_length (float): Focal length of the lens in mm.
#     - working_distance (float): Distance between the camera and the subject in mm.

#     Returns:
#     - projected_width (float): Real-world width of the projected image in mm.
#     - projected_height (float): Real-world height of the projected image in mm.
#     """
#     projected_width = (sensor_width * working_distance) / focal_length
#     projected_height = (sensor_height * working_distance) / focal_length
#     return projected_width, projected_height

# # # Example inputs
# # sensor_width = 36.0  # mm (Full-frame sensor width)
# # sensor_height = 24.0  # mm (Full-frame sensor height)
# # focal_length = 280.0  # mm
# # working_distance = 384431.0  # mm (1 meter)

# # fov_h, fov_v, fov_h_dist, fov_v_dist = calculate_fov(sensor_width, sensor_height, focal_length, working_distance)
# # proj_width, proj_height = calculate_projected_image_size(sensor_width, sensor_height, focal_length, working_distance)

# # print(f"Projected Width: {proj_width:.2f} mm")
# # print(f"Projected Height: {proj_height:.2f} mm")

# # print(f"Horizontal FoV: {fov_h:.2f}°")
# # print(f"Vertical FoV: {fov_v:.2f}°")
# # print(f"Horizontal FoV at {working_distance} mm: {fov_h_dist:.2f} mm")
# # print(f"Vertical FoV at {working_distance} mm: {fov_v_dist:.2f} mm")
