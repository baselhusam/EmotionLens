# EmotionLens

![EmotionLens Logo](assets/logo.png)

## Introduction

EmotionLens is a state-of-the-art system designed for real-time facial emotion recognition and object detection. By leveraging advanced deep learning architectures.

## Features

- Real-time facial emotion recognition using Residual Masking Network
- Object detection with YOLO for enhanced data accuracy
- Tracker for real-time data processing and analysis

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
The face detection model utilizes a convolutional neural network (CNN) architecture to accurately detect and localize faces in real-time video streams. The model has been trained on diverse datasets to ensure robustness across various lighting conditions and environments.

### Emotion Recognition
EmotionLens employs the Residual Masking Network for emotion recognition. This deep learning model is specifically designed to interpret a wide range of emotional states from facial expressions. It categorizes emotions into positive, negative, and neutral, providing nuanced insights into student reactions.

### YOLO Object Detection
The YOLO (You Only Look Once) object detection system is integrated to identify and track objects such as laptops, phones, and other classroom-related items. This addition enhances the context of the emotion recognition system, allowing for more accurate interpretation of student engagement and behaviors.

## Tracker
The tracker module is responsible for processing real-time data from both the face detection and emotion recognition models. It ensures that the system can handle high-frequency video frames, maintaining low latency and high accuracy. The tracker also integrates data from the YOLO model to provide a comprehensive analysis of the classroom environment.

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

3. Object Detection: The YOLO model identifies objects within the classroom to provide context for emotional states.

4. Tracker: Integrates and processes data from all models to deliver real-time insights.

## Configuration
The `config.py` file allows customization of various parameters to tailor the system to specific requirements. Below is an overview of the configurable parameters:

- `SRC_VIDEO`: Video source (0 for webcam or path to video file)

- `FILTER_EMOTIONS`: Filter emotions to be positive, negative, and neutral only (True/False)
Emotion Model Config

- `APPLY_TRACKER`: Apply tracker (True/False)

- `APPLY_YOLO`: Apply YOLO object detection (True/False)

- `YOLO_REQ_CLS`: List of required classes to filter YOLO results

The config.py file ensures that the EmotionLens system is highly customizable to fit various operational requirements and environments.



## License
This project is licensed under the MIT License. See the LICENSE file for more details.

![poster](assets/poster.jpg)


