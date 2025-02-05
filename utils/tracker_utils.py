import numpy as np
from scipy.spatial.distance import cdist
import cv2
import argparse
import os
import torch
import imageio
import logging
import time
from utils.files_utils import *

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

logging.info("Script started")

class Tracker:
    MAX_DISAPPEAR_LIMIT = 5
    SIZE_MATCH_THRESHOLD = 0.3
    
    def __init__(self):
        self.next_unique_id = 0
        self.trackers = {}  # Stores centroids
        self.disappear_trackers = {}  # Tracks disappearance count
        self.tracked_bboxes = {}  # Stores bounding boxes
        self.tracked_class_names = {}  # Stores class names

    def init_object(self, centroid, boxes, class_name):
        self.trackers[self.next_unique_id] = centroid
        self.tracked_bboxes[self.next_unique_id] = boxes
        self.tracked_class_names[self.next_unique_id] = class_name
        self.disappear_trackers[self.next_unique_id] = 0
        self.next_unique_id += 1

    def del_object(self, track_id):
        del self.trackers[track_id]
        del self.tracked_bboxes[track_id]
        del self.disappear_trackers[track_id]
        del self.tracked_class_names[track_id]

    def update_object(self, detections):
        class_names = [detection[0] for detection in detections]
        bboxes = [detection[1:5] for detection in detections]
        if len(bboxes) == 0:
            # Handle disappearance
            for oid in list(self.disappear_trackers.keys()):
                self.disappear_trackers[oid] += 1
                if self.disappear_trackers[oid] > Tracker.MAX_DISAPPEAR_LIMIT:
                    self.del_object(oid)
            return self
        
        else:
            input_centroids = np.zeros((len(bboxes), 2)) 
            bbox_areas = np.zeros(len(bboxes))
            
            for i in range(len(bboxes)):
                x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                cx, cy = x + w / 2, y + h / 2
                input_centroids[i] = (cx, cy)
                bbox_areas[i] = w * h

            if len(self.trackers) == 0:
                for i in range(len(input_centroids)):
                    self.init_object(input_centroids[i], bboxes[i], class_names[i])

            else:
                tracker_centroids = list(self.trackers.values())
                tracker_areas = [self.tracked_bboxes[tid][2] * self.tracked_bboxes[tid][3] 
                                 for tid in self.trackers.keys()]
                distance_matrix = cdist(np.array(tracker_centroids), input_centroids)

                rows = distance_matrix.min(axis=1).argsort()
                cols = distance_matrix.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()

                tracker_ids = list(self.trackers.keys()) 
                for row, col in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue
                    
                    track_id = tracker_ids[row]
                    
                    # Check class name match
                    if self.tracked_class_names[track_id] != class_names[col]:
                        continue
                    
                    # Check bounding box size match
                    tracked_area = tracker_areas[row]
                    detected_area = bbox_areas[col]
                    size_difference = abs(tracked_area - detected_area) / max(tracked_area, detected_area)
                    if size_difference > Tracker.SIZE_MATCH_THRESHOLD:
                        continue  # Skip if the size difference is too large
                    
                    # Update the tracker
                    self.trackers[track_id] = input_centroids[col]
                    self.tracked_bboxes[track_id] = bboxes[col]
                    self.disappear_trackers[track_id] = 0
                    usedRows.add(row)                                 
                    usedCols.add(col)

                unusedRows = set(range(0, distance_matrix.shape[0])).difference(usedRows)
                unusedCols = set(range(0, distance_matrix.shape[1])).difference(usedCols)

                if distance_matrix.shape[0] >= distance_matrix.shape[1]:
                    for r in unusedRows:
                        track_id = tracker_ids[r]
                        self.disappear_trackers[track_id] += 1
                        if self.disappear_trackers[track_id] > Tracker.MAX_DISAPPEAR_LIMIT:
                            self.del_object(track_id)
                else:
                    for c in unusedCols:
                        self.init_object(input_centroids[c], bboxes[c], class_names[c])

        return self




def convert_detection_format(detection):
    """
    Convert detection format from (x1, y1, x2, y2, x3, y3, x4, y4, class_name, confidence) 
    to the original format [class_name, x, y, nw, nh, confidence].
    
    Parameters:
    - detection: tuple of (x1, y1, x2, y2, x3, y3, x4, y4, class_name, confidence)
    
    Returns:
    - A list in the format [class_name, x, y, nw, nh, confidence]
    """
    # Unpack the detection
    x1, y1, x2, y2, x3, y3, x4, y4, class_name, confidence = detection
    
    # Calculate the top-left corner (x, y) and the width and height (nw, nh)
    x = int(min(x1, x2, x3, x4))
    y = int(min(y1, y2, y3, y4))
    
    nw = int(max(x1, x2, x3, x4) - x)
    nh = int(max(y1, y2, y3, y4) - y)
    
    # Return the formatted detection
    return [class_name, x, y, nw, nh, confidence]


def load_detections_from_file(frame_number, folder):
    """Load detections from the specified file."""
    file_name = f"det_fr_{frame_number}.txt"
    file_path = os.path.join(folder, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Detection file {file_path} does not exist.")

    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by commas
            parts = line.strip().split(',')
            
            # Ensure that there are enough components in the line
            if len(parts) >= 10:
                # Extract the bounding box coordinates, class name, and confidence score
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])  # Coordinates
                class_name = parts[8]  # Class name
                confidence = float(parts[9])  # Confidence score
                detection = convert_detection_format((x1, y1, x2, y2, x3, y3, x4, y4, class_name, confidence))
                # Append the detection to the list
                detections.append(detection)
    return detections

def start_tracker(media_path, detection_folder, output_path, tracking_txt_path):
    tracker = Tracker()

    # Initialize video capture
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        logging.info(f"Error opening video file: {media_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.info("Warning: FPS is 0, defaulting to 30")
        fps = 30  # Set to default FPS
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.info(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

    # Initialize the VideoWriter to save the output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    if not out.isOpened():
        logging.info(f"Error: Could not open VideoWriter for the output file {output_path}")
        return

    # Initialize frame count
    ret = True
    frame_count = 0

    while ret:
        ret, frame = cap.read()
        if not ret:
            logging.info(f"Error reading frame {frame_count}. Stopping video processing.")
            break

        frame_number = f"{frame_count:04d}"
        try:
            detections = load_detections_from_file(frame_number, detection_folder)
            # logging.info(f"Detections for frame {frame_number} found: {len(detections)} detections")
        except FileNotFoundError as e:
            logging.info(f"Warning: {e}")
            detections = []

        # Update the tracker with detections
        tracker_res = tracker.update_object(detections)

        # Draw tracking data on the frame
        for id, boxes in tracker_res.tracked_bboxes.items():
            x, y = (int(boxes[0]), int(boxes[1]))
            w, h = (int(boxes[2]), int(boxes[3]))
            cv2.rectangle(frame, (x, y), (x + w, y + h), thickness=2, color=(255, 0, 0))
            cv2.putText(frame, str(id), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Write output tracker text file
        frame_tracker_path = os.path.join(tracking_txt_path, f"track_fr_{frame_number}.txt")
        write_track_file(frame_tracker_path, tracker_res)

        # Write the frame with tracking data to the output video
        out.write(frame)

        # Log progress for debugging
        if frame_count % 100 == 0:
            logging.info(f"Processed {frame_count} frames...")

        # Wait for 30ms and break on 'Esc' key press
        key = cv2.waitKey(30)
        if key == 27:  # Esc key
            logging.info("Esc key pressed, stopping video processing.")
            break

        frame_count += 1
        if frame_count >= 7517:
            logging.info("Reached the end of the video.")
            break
        if frame_count >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            logging.info("Reached the end of the video.")
            break

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    logging.info(f"Video processing completed. Output saved as '{output_path}'.")

    # Verify the output file
    if os.path.exists(output_path):
        logging.info(f"Output video file '{output_path}' exists.")
        try:
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                logging.info(f"Successfully opened the output video '{output_path}' for verification.")
            else:
                logging.info(f"Error: Output video '{output_path}' could not be opened.")
            test_cap.release()
        except Exception as e:
            logging.info(f"Error verifying the output video: {e}")
    else:
        logging.info(f"Error: Output video file '{output_path}' does not exist.")
