# Real_time_face_detection

# Real-Time Face Recognition System

## Overview

This project implements a real-time face recognition system using OpenCV and the face_recognition library. The system detects faces from a video stream, recognizes previously seen faces, and displays metadata about each recognized face. It supports both Raspberry Pi cameras and USB webcams.

## Features

- Real-time face detection and recognition
- Save and load known faces and their metadata
- Display metadata such as first seen time, last seen time, and visit count
- Handles both Raspberry Pi camera modules and USB webcams

## Requirements

- Python 3.x
- OpenCV
- face_recognition library
- NumPy

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/face-recognition-system.git
    cd face-recognition-system
    ```

2. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

3. If you're using a Raspberry Pi camera, ensure you have the necessary libraries for gstreamer installed:

    ```sh
    sudo apt-get install libgstreamer1.0-dev
    ```

## Usage

1. Ensure you have a camera connected to your system.
2. Run the script:

    ```sh
    python face_recognition_system.py
    ```

3. The system will start capturing video and recognizing faces. Press `q` to quit.

## Configuration

- To switch between using a Raspberry Pi camera and a USB webcam, change the `USING_RPI_CAMERA_MODULE` variable at the beginning of the script:

    ```python
    USING_RPI_CAMERA_MODULE = False  # Set to True if using a Raspberry Pi camera module
    ```

## Functions

### `save_known_faces()`
Saves known face encodings and metadata to a file (`known_faces.dat`).

### `load_known_faces()`
Loads known face encodings and metadata from a file (`known_faces.dat`).

### `get_jetson_gstreamer_source(capture_width, capture_height, display_width, display_height, framerate, flip_method)`
Returns a gstreamer pipeline string for capturing video from a Raspberry Pi camera on a Jetson Nano.

### `register_new_face(face_encoding, face_image)`
Registers a new face encoding and its associated metadata.

### `lookup_known_face(face_encoding)`
Looks up a face encoding in the list of known faces and returns its metadata if found.

### `main_loop()`
Main loop for capturing video, detecting faces, recognizing faces, and displaying the results.

## Acknowledgements

This project was developed by Zaryab Rahman. It is based on the face_recognition library and OpenCV.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
