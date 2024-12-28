# from yolo import YoloDetection
import cv2
import argparse
from centroid_tracker import Tracker
import os
import torch
import imageio
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also logging.info to console
    ]
)

logging.info("Script started")

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


def load_detections_from_file(frame_number):
    """Load detections from the specified file."""
    folder = "/Users/selmabenhassine/Desktop/Masters/MA3/SemProj/Galatsi_Dataset_archive/DJI_0763_detection"
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


def start_tracker():
    media_path = "/Users/selmabenhassine/Desktop/Masters/MA3/SemProj/Galatsi_Dataset_archive/DJI_0763.MOV"
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
    output_path = 'tracked_output.mp4'
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
            detections = load_detections_from_file(frame_number)
            logging.info(f"Detections for frame {frame_number} found: {len(detections)} detections")
        except FileNotFoundError as e:
            logging.info(f"Warning: {e}")
            detections = []

        # Update the tracker with detections
        tracker_res = tracker.update_object([x[1:5] for x in detections])

        # Draw tracking data on the frame
        for id, boxes in tracker_res.items():
            x, y = (int(boxes[0]), int(boxes[1]))
            w, h = (int(boxes[2]), int(boxes[3]))
            cv2.rectangle(frame, (x, y), (x + w, y + h), thickness=2, color=(255, 0, 0))
            cv2.putText(frame, str(id), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

start_tracker()