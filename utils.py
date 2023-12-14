import cv2
import numpy as np

def draw_bounding_box(faces, frame):

    """
    Draws bounding boxes around the detected faces in the input image.
    
    Parameters:
    ----------
    faces: list
        A list of tuples containing the bounding boxes coordinates of the detected faces.
    frame: numpy.ndarray
        The input image with RGB color space.

    Returns:
    -------
    frame: numpy.ndarray
        The input image with bounding boxes drawn around the detected faces.
    
    Raises:
    ------
    IOError
        If the input image is not a valid image.

    """

    # Check if the image loaded correctly
    if not isinstance(frame, np.ndarray):
        raise IOError('Input is not a valid image')

    # Loop over all the faces and draw bounding boxes around them
    for face in faces:
        
        # Get the B-Box & Draw it around the faces
        top, right, bottom, left = face
        frame = cv2.rectangle(frame, (left,top) , (right, bottom), (0,255,0), 2)

    return frame


def get_faces(faces, frame):

    """
    Crops the detected faces from the input image.

    Parameters:
    ----------
    faces: list
        A list of tuples containing the bounding boxes coordinates of the detected faces.
    frame: numpy.ndarray
        The input image with RGB color space.

    Returns:
    -------
    face_imgs: list
        A list of numpy.ndarray containing the detected faces.

    Raises:
    ------
    IOError
        If the input image is not a valid image.

    """

    # Check if the image loaded correctly
    if not isinstance(frame, np.ndarray):
        raise IOError('Input is not a valid image')

    # Loop over all the faces and draw bounding boxes around them
    face_imgs = []
    for face in faces:
        
        # Get the coordinates of the bounding box & Extract the face
        top, right, bottom, left = face
        face_img = frame[top:bottom, left:right]
        face_imgs.append(face_img)

    return face_imgs


def preprocess_frame(frames):

    """
    Preprocesses the input frame before feeding it to the model.

    It does the following processing:
    1. Convert to Gray Scale (1 channel)
    2. Resize to 48x48 (Width x Height); the input size of the model
    3. Convert to array 
    4. Expand dimension to 1x48x48x1; the input shape of the model
    5. Normalize the image by dividing it by 255.0 

    Parameters:
    ----------
    frame: list
        A list of numpy.ndarray containing the detected faces.

    Returns:
    -------
    gray: list 
        A list of numpy.ndarray containing the preprocessed faces.
    """

    gray_frames = []
    for frame in frames:

        # Convert to Gray Scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        gray = cv2.resize(gray, (48,48))
        
        # Convert to array
        gray = np.array(gray)
        
        # Expand dimension to 1x48x48x1
        gray = np.expand_dims(gray, axis=0)
        gray = np.expand_dims(gray, axis=-1)
        
        # Normalize
        gray = gray/255.0

        # Append to the list
        gray_frames.append(gray)
    
    return gray_frames


def get_pred(processed_imgs, model, classes):

    """
    Makes prediction using the specified model.

    Parameters:
    ----------
    processed_img: list
        A list of numpy.ndarray containing the preprocessed faces.
    model: keras.engine.training.Model
        The trained model.
    classes: list
        A list of strings containing the names of the classes.

    Returns:
    -------
    pred: list
        A list of strings containing the predicted classes.

    """
    preds = []
    for processed_img in processed_imgs:

        # Make prediction
        pred = model.predict(processed_img)
        pred = np.argmax(pred)
        
        # Get the class name
        cls = classes[pred]
        preds.append(cls)

    return preds

def write_emotions(faces, frame, preds):

    """
    Writes the predicted class above the detected face.

    Parameters:
    ----------
    faces: list
        A list of tuples containing the bounding boxes coordinates of the detected faces.
    frame: numpy.ndarray
        The input image with RGB color space.
    preds: list
        A list of strings containing the predicted classes.

    Returns:
    -------
    frame: numpy.ndarray
        The input image with the predicted classes written above the detected faces.
    """

    for i, face in enumerate(faces):
        frame = cv2.rectangle(frame, (face[3], face[0]-30), (face[1], face[0]), (0,255,0), -1)
        frame = cv2.putText(frame, preds[i].capitalize(), (face[3], face[0]-5), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
    
    return frame

def write_fps(frame, fps):

    """
    Writes the FPS on the top left corner of the frame.

    Parameters:
    ----------
    frame: numpy.ndarray
        The input image with RGB color space.
    fps: float
        The FPS of the video.

    Returns:
    -------
    frame: numpy.ndarray
        The input image with the FPS written on the top left corner.
    """

    # Write the FPS on the top left corner
    frame = cv2.putText(frame, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame