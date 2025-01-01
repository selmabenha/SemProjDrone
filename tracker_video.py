import cv2
import os
import logging

# logging.basicConfig(
#     level=logging.DEBUG,  # Set the minimum level of messages to capture
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("logs/script.log"),  # Write logs to this file
#         logging.StreamHandler()  # Optionally, also log to console
#     ]
# )
reference_frames = [1, 300, 550, 800, 1050, 1300, 1550, 1800, 2050, 2350, 2600, 2900, 3200, 3500, 3800, 4100, 4350, 4600, 4900, 5150, 5400, 5650, 6050, 6300, 6550, 6800, 7517]

def load_tracking_from_file(frame_number):
    """Load trackers from the specified file."""
    folder = "/Users/selmabenhassine/Desktop/SemProjDrone/DJI_tracking_short"
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
    # map_path = "/Users/selmabenhassine/Desktop/SemProjDrone/extracted_frames"
    map_path = "/Users/selmabenhassine/Desktop/SemProjDrone/output/images/base_out/final_stitched_image_05.jpg"
    # Load the map image to get frame width and height
    # map_image = cv2.imread(f"{map_path}/frame_0001.jpg")
    map_image = cv2.imread(map_path)
    map_image_index = 1
    if map_image is None:
        print(f"Error loading map image: {map_path}")
        return
    
    frame_width, frame_height = map_image.shape[1], map_image.shape[0]
    fps = 30  # Set a default FPS
    
    print(f"Using map image properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

    # Initialize the VideoWriter to save the output
    output_path = '/Users/selmabenhassine/Desktop/SemProjDrone/output/videos/tracked_pix_xy_05.mov'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for the output file {output_path}")
        return

    # Initialize frame count
    frame_count = 0

    while True:
        frame_number = f"{frame_count:04d}"
        if frame_count == 1717: 
            print("Done")
            break
        # if frame_count != 1 and frame_count == reference_frames[map_image_index]:
        #     new_map_path = f"{map_path}/frame_{frame_number}.jpg"
        #     map_image = cv2.imread(new_map_path)
        #     map_image_index += 1
        #     print(f"new map image path is {new_map_path}")
        try:
            trackers = load_tracking_from_file(frame_number)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
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
            print(f"Processed {frame_count} frames...")

        # Wait for 30ms and break on 'Esc' key press
        key = cv2.waitKey(30)
        if key == 27:  # Esc key
            print("Esc key pressed, stopping video processing.")
            break

        frame_count += 1

        # Stop if there are no more trackers for further frames
        if len(trackers) == 0:
            print("No trackers for this frame, stopping video processing.")
            break

    # Release resources
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing completed. Output saved as '{output_path}'.")

    # Verify the output file
    if os.path.exists(output_path):
        print(f"Output video file '{output_path}' exists.")
        try:
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                print(f"Successfully opened the output video '{output_path}' for verification.")
            else:
                print(f"Error: Output video '{output_path}' could not be opened.")
            test_cap.release()
        except Exception as e:
            print(f"Error verifying the output video: {e}")
    else:
        print(f"Error: Output video file '{output_path}' does not exist.")

generate_trackers_video()
