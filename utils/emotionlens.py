import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models import resmasking_dropout1

from retinaface import RetinaFace
from batch_face import RetinaFace as ret
from collections import Counter
import pyshine as ps


import config
import datetime
# from utils.sort.sort import Sort
# from utils.database import get_sec_id

class EmotionLens():
    """
    EmotionLens is a class that uses the Microsoft Azure Emotion API to analyze
    the emotions of a given image.
    """
    def __init__(self):
        """
        Initializes the EmotionLens object.
        """
        self.transform = transforms.Compose(transforms=[transforms.ToPILImage(),
                                                        transforms.ToTensor()])
        self.is_cuda = torch.cuda.is_available()
        self.model = self.get_emo_model()

    def get_emo_model(self):
        """
        Returns the emotion model.

        Returns:
        --------
        emo_model: torch.nn.Module
            The emotion model.
        """
        emo_model = resmasking_dropout1(in_channels=3, num_classes=7)
        if self.is_cuda:
            emo_model.cuda(0)
        state = torch.load("pretrained_ckpt", map_location="cuda")
        emo_model.load_state_dict(state["net"])
        emo_model.eval()
        return emo_model

    def get_faces(self, frame):
        """
        Returns the bounding box of the face in the frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to detect the face in.

        Returns:
        --------
        box: np.ndarray
            The bounding box of the face.
        """  
        detector = ret(gpu_id=0)
        faces = detector(frame, cv=False)
        if faces is None:
            return []
        if config.DEBUG:
            print(f"Num of Faces: {len(faces)}")
        bbox = []
        for face in faces:
            bbox.append(face[0])
        return np.array(bbox)
        

    def get_faces2(self, frame):
        """
        Returns a list of faces' bounding boxes in the frame.
        If debug is True, it prints the number of faces detected.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to detect faces in.
        debug: bool
            If True, it prints the number of faces detected.

        Returns:
        --------
        bbox: np.ndarray
            A numpy array of shape (num_of_faces, 4) where each row is a bounding box of a face.
        """
        face_results = RetinaFace.detect_faces(frame)
        if not isinstance(face_results, dict):
            return []
        if config.DEBUG:
            print(f"Num of Faces: {len(face_results)}")

        bbox = []
        for _, face in face_results.items():
            xmin, ymin, xmax, ymax = face["facial_area"]
            bbox.append([xmin, ymin, xmax, ymax])

        return np.array(bbox)
    
    def ensure_color(self, image):
        """
        Ensures that the image is colored.

        Parameters:
        -----------
        image: np.ndarray
            The image to ensure that it is colored.

        Returns:
        --------
        image: np.ndarray
            The colored image.
        """
        if len(image.shape) == 2:
            return np.dstack([image] * 3)
        elif image.shape[2] == 1:
            return np.dstack([image] * 3)
        return image
    
    def preprocess_face(self, face):
        """
        Preprocesses the face to be fed into the emotion model.

        Parameters:
        -----------
        face: np.ndarray
            The face to preprocess.

        Returns:
        --------
        face: torch.Tensor
            The preprocessed face.
        """
        face = self.ensure_color(face)
        face = cv2.resize(face, config.IMG_SIZE)
        face = self.transform(face)
        face = face.unsqueeze(0)
        if self.is_cuda:
            face = face.cuda(0)
        return face

    def crop_faces(self, frame, faces):
        """
        Returns a list of cropped faces from the frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to crop the faces from.
        faces: np.ndarray
            A numpy array of shape (num_of_faces, 4) where each row is a bounding box of a face.

        Returns:
        --------
        cropped_faces: List[np.ndarray]
            A list of cropped faces.
        """
        cropped_faces = []
        for xmin, ymin, xmax, ymax in faces:
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cropped_faces.append(frame[ymin:ymax, xmin:xmax])
        return cropped_faces
    

    @torch.no_grad()
    def get_emotions(self, frame, faces):
        """
        Returns a list of emotions of the faces.

        Parameters:
        -----------
        faces: List[np.ndarray]
            A list of cropped faces.

        Returns:
        --------
        emotions: List[str]
            A list of emotions of the faces.
        """
        emotions = []
        faces = self.crop_faces(frame, faces)
        for face in faces:
            face = self.preprocess_face(face)
            out = self.model(face)
            emo = torch.argmax(out, dim=1)
            emotion = config.FER_2013_EMO_DICT[emo.item()]
            emotions.append(emotion)

        emotions = self.filter_emotions(emotions)
        return emotions
    
    def draw_emotion(self, frame, face, emotion):
        """
        Draws the emotion on the face.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to draw the emotion on.
        face: np.ndarray
            The face to draw the emotion on.
        emotion: str
            The emotion to draw.
        """
        try:
            self.draw_faces(frame, face)
            self.write_emotion(frame, face, emotion)
        except Exception as e:
            print(f"Error in drawing emotion: {e}")
    
    def write_emotion(self, frame, face, emotion):
        """
        Writes the emotion on the face.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to write the emotion on.
        face: np.ndarray
            The face to write the emotion on.
        emotion: str
            The emotion to write.
        """
        try:
            for i, f in enumerate(face):
                f = [int(i) for i in f]
                x1, y1, x2, y2 = f
                text = emotion[i]
                ps.putBText(frame,
                            text,
                            text_offset_x=x1-10,
                            text_offset_y=y1-10,
                            thickness=1,
                            vspace=2,
                            hspace=2, 
                            font_scale=0.6,
                            background_RGB=(0,250,250),
                            text_RGB=(255,250,250))
        except Exception as e:
            print(f"Error in writing emotion: {e}")
        
    def draw_faces(self, frame, faces, r=4, d=2, color=(255,255,127)):
        """
        Draws the faces on the frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to draw the faces on.
        faces: List[np.ndarray]
            A list of faces to draw.
        """
        for face in faces:
            # Convert the face to int
            face = [int(i) for i in face]
            pt1 = (face[0], face[1])
            pt2 = (face[2], face[3])
            # color = (255,255,127)
            thickness = 1
            # r = 4  # Change this value as needed
            # d = 2  # Change this value as needed

            x1,y1 = pt1
            x2,y2 = pt2

            # Top left
            cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
            cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
            cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

            # Top right
            cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
            cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
            cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

            # Bottom left
            cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
            cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
            cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

            # Bottom right
            cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
            cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
            cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    
    def filter_emotions(self, emotions):
        """
        Process the emotions to "Positive", "Negative", and "Neutral".

        Parameters:
        -----------
        emotions: List[str]
            A list of emotions of the faces.

        Returns:
        --------
        emotions: List[str]
            A list of emotions of the faces.
        """
        positive_emotions = ["happy", "surprise"]
        negative_emotions = ["angry", "disgust", "fear", "sad"]
        neutral_emotions = ["neutral"]

        for i, emo in enumerate(emotions):
            if emo in positive_emotions:
                emotions[i] = "Positive"
            elif emo in negative_emotions:
                emotions[i] = "Negative"
            elif emo in neutral_emotions:
                emotions[i] = "Neutral"
        return emotions
    
    def draw_emotion_counter(self, frame, emotions_counter):
        """
        Draws the emotion counter on the frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to draw the emotion counter on.
        emotions_counter: Dict[str, int]
            A dictionary of emotions and their counts.
        """
        
        for i, (emotion, count) in enumerate(emotions_counter.items()):
            text = f"{emotion}: {count}"
            try:
                ps.putBText(frame,
                            text,
                            text_offset_x=10,
                            text_offset_y=(i+1)*20,
                            thickness=1,
                            vspace=2,
                            hspace=2, 
                            font_scale=0.6,
                            background_RGB=(0,250,250),
                            text_RGB=(255,250,250))
            except Exception as e:
                print(f"Error in drawing emotion counter: {e}")
            
    def track(self, frame, faces, track_dict):
        """
        Tracks the faces in the frame.

        Parameters:
        -----------
        frame: np.ndarray
            The frame to track the faces in.
        faces: List[np.ndarray]
            A list of faces to track.

        Returns:
        --------
        frame: np.ndarray
            The frame with the faces tracked.
        """
        mot = Sort()
        detections = []
        for face in faces:
            x1, y1, x2, y2 = face
            detections.append([x1, y1, x2, y2, 1])
        detections = np.array(detections)
        
        tracked_faces = mot.update(detections)
        for x1, y1, x2, y2, face_id in tracked_faces:
            
            # Get Face Emotion
            emotion = self.get_emotions(frame, [x1, y1, x2, y2])

            face_obj = track_dict.get(face_id, None)
            if face_obj is None:
                track_dict[face_id] = []

            track_dict[face_id].append(emotion)

        return track_dict
    
    def reset_track_dict(self, track_dict, emotions_counter):
        """
        Resets the track dictionary and adds the emotion counter to it.

        Parameters:
        -----------
        track_dict: Dict[int, List[str]]
            The track dictionary.
        emotions_counter: Dict[str, int]
            A dictionary of emotions and their counts.

        Returns:
        --------
        track_dict: Dict[int, List[str]]
            The reset track dictionary.
        emotions_counter: Dict[str, int]
            The emotion counter.
        """
        for face_id, emotions in track_dict.items():
            counter = Counter(emotions)
            for emotion, count in counter.items():
                emotions_counter[emotion] += count
        return {}, emotions_counter
    
    def _make_detection(self, faces):
        """
        Makes a detection from the faces.

        Parameters:
        -----------
        faces: List[np.ndarray]
            A list of faces.

        Returns:
        --------
        detection: Dict[int, List[int]]
            A dictionary of face id and the face bounding box.
        """
        detections = []
        for face in faces:
            x1, y1, x2, y2 = face
            detections.append([x1, y1, x2, y2, 1])
        detections = np.array(detections)
        return detections

        
    def apply_tracker(self, mot, faces, emotions, track_dict):
        detection = self._make_detection(faces)
        tracked_faces = mot.update(detection)
        for (x1, y1, x2, y2, face_id), emotion in zip(tracked_faces, emotions):
            face_obj = track_dict.get(face_id, None)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if face_obj is None:
                track_dict[face_id] = {"emotions": [],
                                       "face": [x1, y1, x2, y2],
                                       "alive": False}
            track_dict[face_id]["emotions"].append(emotion)
            track_dict[face_id]["face"] = [x1, y1, x2, y2]
            track_dict[face_id]["alive"] = True

        return track_dict
    
    def count_emotions(self, track_dict):
        """
        Counts the emotions in the track dictionary.

        Parameters:
        -----------
        track_dict: Dict[int, Dict[str, List[str]]]
            The track dictionary.
        emotions_counter: Dict[str, int]
            A dictionary of emotions and their counts.

        Returns:
        --------
        emotions_counter: Dict[str, int]
            The emotion counter.
        """
        emotions_counter = {"Positive": 0,
                            "Negative": 0,
                            "Neutral": 0}
        for face_id, face_obj in track_dict.items():
            emotions = face_obj["emotions"]
            counter = Counter(emotions)
            maj_vote_emotion = counter.most_common(1)[0][0]
            emotions_counter[maj_vote_emotion] += 1

        return emotions_counter
    
    def update_firebase_data(self, firebase_data, emotions_counter):
        """
        Updates the firebase data with the emotions counter.

        Parameters:
        -----------
        firebase_data: Dict[str, Union[int, List]]
            The firebase data.
        emotions_counter: Dict[str, int]
            A dictionary of emotions and their counts.

        Returns:
        --------
        firebase_data: Dict[str, Union[int, List]]
            The updated firebase data.
        """
        for emotion, count in emotions_counter.items():
            firebase_data[emotion] += count
            firebase_data[f"OverAll{emotion}"].append(count)
        return firebase_data
    
    def send_data_to_firebase(self, firebase_data, db, lec_name, id_sec_num):
        """
        Sends the data to the Firebase.

        Parameters:
        -----------
        firebase_data: Dict[str, Union[int, List]]
            The firebase data.
        db: firebase_admin.firestore.client.Client
            The Firebase client.
        lec_name: str
            The lecturer name.
        id_sec_num: str
            The ID, section, and number of the student.
        
        Returns:
        --------
        ret: bool
            True if the data was sent successfully, False otherwise.
        """
        sent_data = firebase_data.copy()
        section_reference = db.collection("Sections").document(get_sec_id(db, idx=4))
        sent_data["LectureID_SectionNum"] = section_reference
        sent_data["Date"] = datetime.datetime.now() 

        all_emotions = ["Positive", "Negative", "Neutral"]
        all_overall = [f"OverAll{emotion}" for emotion in all_emotions]
        sums_overall = sum([sent_data[in_over][-1] for in_over in all_overall])

        for overall in all_overall:
            sent_data[overall] = sent_data[overall][-1]
            sent_data[overall] /= sums_overall
            sent_data[overall] = round(sent_data[overall], 3)

        try:
            # doc_ref = db.collection(config.DB_CONFIG["collection_name"]).document(config.DB_CONFIG["doc_reference"][lec_name])
            doc_ref = db.collection(config.DB_CONFIG["collection_name"]).document()
            doc_ref.create(sent_data)
            return True
        except Exception as e:
            print(f"Logging: There was an error in sending the data to Firebase: {e}")
            return False
        
    def filter_yolo_results(self, results, labels, frame):
        """
        Filters the YOLO results to only include the faces.

        Parameters:
        -----------
        results: List[YOLOResult]
            The YOLO results.

        Returns:
        --------
        faces: List[YOLOResult]
            The faces.
        """
        boxes = [res.boxes.xyxy for res in results][0]
        clss = [res.boxes.cls for res in results][0]
        idx = [i for i, cls in enumerate(clss) if labels[int(cls)] in config.YOLO_REQ_CLS]
        boxes = [boxes[i] for i in idx]
        clss = [clss[i] for i in idx]

        if config.DRAW:
            for box, cls in zip(boxes, clss):
                self.draw_faces(frame, [box], r=20, d=15, color=(255,150,250))
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                try:
                    ps.putBText(frame, 
                                labels[int(cls)], 
                                text_offset_x=x1, 
                                text_offset_y=y2-20, 
                                vspace=2, 
                                hspace=2, 
                                font_scale=0.4, 
                                background_RGB=(255,150,250), 
                                text_RGB=(255,150,250))
                except Exception as e:
                    print(f"Error in drawing faces: {e}")
        return boxes

