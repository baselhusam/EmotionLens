# EmotionLens

![EmotionLens Logo](https://github.com/baselhusam/EmotionLens/blob/master/assets/logo.png?raw=true)

## Introduction

EmotionLens is a state-of-the-art system designed for real-time facial emotion recognition and object detection. By leveraging advanced deep learning architectures.

## Features

- Real-time facial emotion recognition using Residual Masking Network. [used from this repo](https://github.com/phamquiluan/ResidualMaskingNetwork)
- Object detection with YOLO for enhanced Emotion Results
- Tracker for tracking faces for enhancing the Emotion Results

## Project Structure

```plaintext
EmotionLens/
├── assets/
│   ├── logo.png
│   ├── poster.jpg
├── utils/
│   ├── emotionlens.py
│   ├── sort.py
│   ├── general.py
├── config.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Models

### Face Detection
The face detection model For detecting faces. The used Model is `RetinaFace` in batch mode for handling large number of faces in Real-Time mode.

### Emotion Recognition
EmotionLens employs the Residual Masking Network for emotion recognition. This deep learning model is specifically designed to interpret a wide range of emotional states from facial expressions. It categorizes emotions into angry, disgust, fear, happy, sad, surprise, and neutral.

### YOLO Object Detection
The YOLO (You Only Look Once) object detection system is integrated to identify objects such as laptops, phones, and other related items. This addition enhances the context of the emotion recognition system, allowing for more accurate interpretation of engagement and behaviors.

## Tracker
The tracker module is responsible for tracking real-time faces from the face detection. This is for tacking the majority vote for each object as its final emotion. The used tracking algorithm is SORT (Simple Online Real-Time Tracking) [from the following repo](https://github.com/abewley/sort)

## Setup Instructions
1. Clonse the Repository:
```sh
git clone https://github.com/baselhusam/EmotionLens.git
cd EmotionLens
``` 
2. Install the necessary dependencies:
```sh
pip install -r requirements.txt
```
3. Run the application:
```sh
python main.py
```

## Usage
1. Face Detection: The model processes video frames to detect and localize faces.

2. Emotion Recognition: Detected faces are analyzed to recognize and categorize emotions.

3. Object Detection: The YOLO model identifies objects to provide context for emotional states.

## Configuration
The `config.py` file allows customization of various parameters to tailor the system to specific requirements. Below is an overview of the configurable parameters:

- `SRC_VIDEO`: Video source (0 for webcam or path to video file)

- `FILTER_EMOTIONS`: Filter emotions to be positive, negative, and neutral only (True/False)

- `APPLY_TRACKER`: Apply tracker (True/False)

- `APPLY_YOLO`: Apply YOLO object detection (True/False)

- `YOLO_REQ_CLS`: List of required classes to filter YOLO results

The `config.py` file ensures that the EmotionLens system is highly customizable to fit various operational requirements and environments.



## Poster
![alt text](https://github.com/baselhusam/EmotionLens/blob/master/assets/Poster.jpg?raw=true)


