"""
This is the Starting point of the project, EmotionLens.

Created on Sat Mar 23 2024
@author: Basel Husam
"""
import cv2
from typing import Dict, List
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, firestore

from utils.emotionlens import EmotionLens
from utils.general import get_times_and_bounds
from utils.sort.sort import Sort
import config



def main():
    """
    The main function of the project.
    """
    # Initialize the EmotionLens, YOLO, and SORT
    emotionlens = EmotionLens()
    yolo = YOLO(config.YOLO_VERSION)
    mot = Sort()

    # Initialize the Firebase
    cred = credentials.Certificate(config.DB_CONFIG["cred_path"])
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Initialize the video capture
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if config.REALTIME:
        cap = cv2.VideoCapture(0)
    bounds, times = get_times_and_bounds(cap)
    if not config.PARTITION:
        if config.REALTIME:
            times = list(range(1, 10_000))
        else:
            times = list(range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1))

    # Initialize the variables
    frame_counter = 0
    track_dict: Dict[int, Dict[List, List]] = {}
    emotions_counter = config.EMOTION_COUNTER
    firebase_data = config.FIREBASE_DATA

    # If Saving the results, initialize the video writer
    if config.SAVE_RESULTS:
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4)) 
        size = (frame_width, frame_height) 
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter("final_out.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (1920, 1080))

    # Start the loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame_counter += 1

        # Get the faces and emotions
        if frame_counter in times:
            faces = emotionlens.get_faces(frame)
            emotions = emotionlens.get_emotions(frame, faces)

            # Apply the YOLO
            yolo_results = yolo(frame, conf=0.5)
            yolo_labels = yolo.names
            ret = emotionlens.filter_yolo_results(yolo_results, yolo_labels, frame)

            # Apply the tracker
            if config.TRACK:
                track_dict = emotionlens.apply_tracker(mot, faces, emotions, track_dict)
                emotions_counter = emotionlens.count_emotions(track_dict)

                # Update the Firebase data
                if frame_counter in bounds and config.SEND_TO_DB:
                    firebase_data = emotionlens.update_firebase_data(firebase_data, emotions_counter)
                    ret = emotionlens.send_data_to_firebase(firebase_data, db, config.DB_CONFIG["lec_name"], config.DB_CONFIG["lec_id_sec_num"])
                    if not ret:
                        print("Error in sending data to Firebase.")
                        break

            # Draw the Faces and Emotions
            if config.DRAW:
                emotionlens.draw_emotion(frame, faces, emotions)
                if config.SAVE_RESULTS:
                    out.write(frame)     
                    print(f"Frame {frame_counter} has been written.")           

        # Draw the emotion counter
        if config.DRAW:
            emotionlens.draw_emotion_counter(frame, emotions_counter)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the video writer
    if config.SAVE_RESULTS:
        out.release()

    # Release the video capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
