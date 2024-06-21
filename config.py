# APP CONFIG
DEBUG = True
# SRC_VIDEO = 0
SRC_VIDEO = "./assets/demo_video.MOV"
FILTER_EMOTIONS = False


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
RESIZE_INIT_FRAME_RATIO = 1
RESIZE_SHOW_FRAME_RATIO = 1


# YOLO CONFIG
YOLO_VERSION = None
