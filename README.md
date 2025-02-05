# File Summaries & User Guide

## Key Files and Their Purposes:
- **lightglue**: Cloned folder from [Lindenberger et al. (2023)](https://arxiv.org/abs/2301.04714) for feature extraction and matching.
- **requires.txt**: Lists necessary dependencies. Run `pip install -r requires.txt` before execution.

## Utility Files:
- **utils/disp_utils.py**: Visualizes and saves keypoint matches, transformations, and stitched results.
- **utils/files_utils.py**: Processes image frames and bounding boxes for transformations and saves data.
- **utils/stitching_utils.py**: Matches keypoints, applies transformations, and stitches images.
- **utils/tracker_utils.py**: Implements object tracker based on bounding boxes and saves results.
- **utils/transform_utils.py**: Adjusts tracked bounding boxes for final stitched image alignment.

## Image Stitching:
- **extract_frames.py**: Extracts frames from video, compares them for keypoint matches, and removes low-overlap frames.  
  - **Input**: `DJI_0763.MOV`  
  - **Output**: `extracted_frames` folder.
- **VideoStitching.py**: Uses extracted frames to produce a stitched image and logs transformation matrices.  
  - **Input**: `extracted_frames`  
  - **Output**: `final_stitched_image.jpg`, `stitching_log.txt`.

## Object Tracking:
- **tracker_video**: Tracks bounding boxes in video, saves tracking information, and generates tracking videos  
  - **Input**: `DJI_0763_detection`, `DJI_0763.MOV`, `final_stitched_image.jpg`, `stitching_log.txt`
  - **Output**: Tracking videos and text files in the `DJI_0763_tracking` folder, `tracked_original_video.mp4`, `tracked_frame_video.mp4`, `tracked_map_video.mp4`

## Runbatch (SCITAS-specific):
- **runbatch/extract.run**: Executes frame extraction.  
- **runbatch/stitching.run**: Executes video stitching.  
- **runbatch/track_video.run**: Executes tracking video generation.

## Notes:
- Ensure accurate file paths in `runbatch` for SCITAS execution.
- For local use, run files manually in order of: extract_frames.py, VideoStitching.py, and tracker_video.py
- SCITAS provides GPU access, but downloaded videos may need processing locally.
