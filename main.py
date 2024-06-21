"""
This is the Starting point of the project, EmotionLens.

Created on Sat Mar 23 2024
@author: Basel Husam
"""
import cv2
from typing import Dict, List, Union

from utils.general import get_emotion_counter
from utils.emotionlens import EmotionLens
from utils.sort import Sort
import config

TRACK_DICT: Dict[int, Dict[str, Union[List, bool]]] = {}
EMOTION_COUNTER = get_emotion_counter(config.FILTER_EMOTIONS)


if __name__=="__main__":

    # Init...
    el = EmotionLens()
    mot = Sort()
    cap = cv2.VideoCapture(config.SRC_VIDEO)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Init Frame Size
        init_resize_ratio = config.RESIZE_INIT_FRAME_RATIO
        frame = cv2.resize(frame, None, fx=init_resize_ratio, fy=init_resize_ratio)

        # Get Faces and Emotions
        faces = el.get_faces(frame)
        emotions = el.get_emotions(frame, faces)

        # Tracker...
        if config.APPLY_TRACKER:
            TRACK_DICT = el.apply_tracker(mot, faces, emotions, TRACK_DICT)
            EMOTION_COUNTER = el.count_emotions(TRACK_DICT)

        # Draw...
        if config.DRAW_EMOTION:
            el.draw_emotion(frame, faces, emotions)
        if config.DRAW_COUNT_EMOTION:
            el.draw_emotion_counter(frame, EMOTION_COUNTER)

        # Show Frame
        show_resize_ratio = config.RESIZE_SHOW_FRAME_RATIO
        show_frame = cv2.resize(frame, None, fx=show_resize_ratio, fy=show_resize_ratio)
        cv2.imshow("Frame", show_frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # Release...
    cap.release()
    cv2.destroyAllWindows()
