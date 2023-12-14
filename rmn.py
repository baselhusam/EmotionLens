
import cv2
from ResidualMaskingNetwork.rmn import RMN
fer = RMN()
fer.video_demo()
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if ret:

        res = fer.detect_emotion_for_single_frame(frame)
        frame = res.draw(frame, res)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()