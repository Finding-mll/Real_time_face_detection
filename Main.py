import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle

# Set this depending on your camera type:
# - True = Raspberry Pi 2.x camera module
# - False = USB webcam or other USB video input (like an HDMI capture device)
USING_RPI_CAMERA_MODULE = False

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []

def save_known_faces():
    """
    Save known faces and their metadata to disk.
    """
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")

def load_known_faces():
    """
    Load known faces and their metadata from disk.
    """
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the RPI camera on a Jetson Nano.
    """
    return (
        f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
        f'width=(int){capture_width}, height=(int){capture_height}, ' +
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
        f'nvvidconv flip-method={flip_method} ! ' +
        f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
        'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    )

def register_new_face(face_encoding, face_image):
    """
    Add a new person to our list of known faces.
    """
    known_face_encodings.append(face_encoding)
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
    })

def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list.
    """
    metadata = None

    if len(known_face_encodings) == 0:
        return metadata

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < 0.65:
        metadata = known_face_metadata[best_match_index]
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 1

    return metadata

def main_loop():
    """
    Main loop to process video from the camera and perform face recognition.
    """
    if USING_RPI_CAMERA_MODULE:
        video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        video_capture = cv2.VideoCapture(0)

    number_of_faces_since_save = 0

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            metadata = lookup_known_face(face_encoding)

            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = f"At door {int(time_at_door.total_seconds())}s"
            else:
                face_label = "New visitor!"
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))
                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)

        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                visits = metadata['seen_count']
                visit_label = f"{visits} visits"
                if visits == 1:
                    visit_label = "First visit"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors at Door", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_known_faces()
    main_loop()

# Author: Zaryab Rahman
# This code is based on the face recognition library and OpenCV.
