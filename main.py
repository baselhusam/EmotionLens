import cv2, os
import numpy as np
import face_recognition

from keras.models import load_model
from utils import draw_bounding_box, get_faces, preprocess_frame, \
                  get_pred, write_emotions, write_fps

# Load the Model with its classes
model = load_model('models/custom.h5')
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Read Video using OpenCV
video_name = "video2.mp4"
vid_path = os.path.join("assets", video_name)
vid = cv2.VideoCapture(0)

SHOW_PREPROCESSED = False
SHOW_FACE = False
FPS = True


while True:

    ret, frame = vid.read()
    if ret:

        # Detect, Draw, and Crop Faces
        faces = face_recognition.face_locations(frame)
        frame = draw_bounding_box(faces, frame)
        face_imgs = get_faces(faces, frame)
        
        # Preprocess and Make Predictions
        processed_imgs = preprocess_frame(face_imgs)
        preds = get_pred(processed_imgs, model, classes)
        frame = write_emotions(faces, frame, preds)

        if SHOW_FACE:
            for i, face_img in enumerate(face_imgs):
                face_img = cv2.resize(face_img, None, fx=2, fy=2)
                cv2.imshow(f'Face {i+1}', face_img)

        if SHOW_PREPROCESSED:
            for i, processed_img in enumerate(processed_imgs):
                cv2.imshow(f'Preprocessed {i+1}', processed_img[0])

        if FPS:
            frame = write_fps(frame, vid.get(cv2.CAP_PROP_FPS))

        # Display the resulting frame
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()