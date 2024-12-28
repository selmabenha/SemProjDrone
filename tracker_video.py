import cv2
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level of messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),  # Write logs to this file
        logging.StreamHandler()  # Optionally, also log to console
    ]
)

def load_tracking_from_file(frame_number):
    """Load trackers from the specified file."""
    folder = "/home/finette/VideoStitching/transform_small"
    file_name = f"track_fr_{frame_number}.txt"
    file_path = os.path.join(folder, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tracking file {file_path} does not exist.")

    trackers = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by commas
            parts = line.strip().split(',')
            
            # Ensure that there are enough components in the line
            if len(parts) >= 10:
                # Extract the bounding box coordinates, class name, and confidence score
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])  # Coordinates
                class_name = parts[8]  # Class name
                class_id = float(parts[9])  # Class ID
                tracker = [x1, y1, x2, y2, x3, y3, x4, y4, class_name, class_id]
                # Append the tracker to the list
                trackers.append(tracker)
    return trackers

def generate_trackers_video():
    map_path = "/home/finette/VideoStitching/selma/output/images/test_transform/final_stitched_image.jpg"
    # Load the map image to get frame width and height
    map_image = cv2.imread(map_path)
    if map_image is None:
        logging.info(f"Error loading map image: {map_path}")
        return
    
    frame_width, frame_height = map_image.shape[1], map_image.shape[0]
    fps = 30  # Set a default FPS
    
    logging.info(f"Using map image properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

    # Initialize the VideoWriter to save the output
    output_path = '/home/finette/VideoStitching/selma/output/tracked_output_mini_transform_small.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    if not out.isOpened():
        logging.info(f"Error: Could not open VideoWriter for the output file {output_path}")
        return

    # Initialize frame count
    frame_count = 0

    while True:
        frame_number = f"{frame_count:04d}"
        try:
            trackers = load_tracking_from_file(frame_number)
            logging.info(f"trackers for frame {frame_number} found: {len(trackers)} trackers")
        except FileNotFoundError as e:
            logging.info(f"Warning: {e}")
            trackers = []

        # Draw tracking data on the frame
        frame = map_image.copy()  # Use a copy of the map image for each frame
        for tracker in trackers:
            # Assuming tracker is a tuple (x1, y1, x2, y2, x3, y3, x4, y4, class_name, confidence)
            x1, y1, x2, y2, x3, y3, x4, y4, class_name, class_id = tracker
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x3), int(y3)), thickness=2, color=(255, 0, 0))
            cv2.putText(frame, f"{class_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

        # Stop if there are no more trackers for further frames
        if len(trackers) == 0:
            logging.info("No trackers for this frame, stopping video processing.")
            break

    # Release resources
    out.release()
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

generate_trackers_video()
