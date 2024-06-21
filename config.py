"""
This file contains all the configuration parameters for the application.
"""
# APP CONFIG
DEBUG = True
SRC_VIDEO = 0 # Video Source (0 for Webcam) or Path to Video File
FILTER_EMOTIONS = False # Filter Emotion to be Positive, Negative, and Neutral Only


# EMOTION MODEL CONFIG
IMG_SIZE = (224, 224)
FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


# TRACKER CONFIG
APPLY_TRACKER = True


# SHOW CONFIG
DRAW_EMOTION = True
DRAW_COUNT_EMOTION = True
RESIZE_INIT_FRAME_RATIO = 1 # Resize Ratio for Initial Frame (1 for Original Size)
RESIZE_SHOW_FRAME_RATIO = 1 # Resize Ratio for Show Frame - Only for Display (1 for Original Size)


# YOLO CONFIG
APPLY_YOLO = True
DRAW_YOLO = True
YOLO_CONF = 0.5
YOLO_VERSION = "YOLOv5n.pt".lower()
YOLO_REQ_CLS = ["person", "cell-phone", "laptop"] # Filter YOLO Results by Required Classes
