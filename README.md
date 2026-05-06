# Feature-Based Object Detection System

<p align="left">
  <img src="https://github.com/user-attachments/assets/d2f4dcaf-1c83-4982-b34b-586802afe0d0" width="40%">
  <img src="https://github.com/user-attachments/assets/f8dc9a12-1b54-4745-82bb-f9899249e288" width="40%">
</p>

## Features

- SIFT feature extraction and matching
- FLANN-based keypoint matching
- Homography estimation using RANSAC
- Bounding box localization
- Multi-object scene detection
- Automated annotation and output generation

## Requirements

- Python
- OpenCV
- NumPy

## Detection Pipeline

1. Extract SIFT keypoints from object images
2. Extract scene keypoints
3. Match descriptors using FLANN
4. Filter matches using Lowe’s Ratio Test
5. Estimate homography using RANSAC
6. Draw bounding boxes on detected objects
