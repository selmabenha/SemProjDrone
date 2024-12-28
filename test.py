import numpy as np
import cv2
import os

# Create a half white, half black image
height, width = 400, 800  # Image dimensions
half_width = width // 2

# Create a black image
image1 = np.zeros((height, width, 3), dtype=np.uint8)
image2 = image1
# Fill the right half with white
image1[:, half_width:] = [255, 255, 255]
image2[:, half_width:] = [200, 200, 200]

output_path = '/home/finette/VideoStitching/selma/output_images'
output_path1 = os.path.join(output_path, f"test1.jpg")
output_path2 = os.path.join(output_path, f"test2.jpg")

cv2.imwrite(output_path1, image1)
cv2.imwrite(output_path2, image2)