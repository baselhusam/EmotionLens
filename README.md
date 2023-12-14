# EmotionLens

Welcome to the EmotionLens project! This repository contains the codebase for our emotion detection model.

## Table of Contents
1. Introduction
2. Installation
3. Usage
4. Model
5. Contributing

## Introduction
EmotionLens is a state-of-the-art emotion detection system that uses deep learning to recognize human emotions from facial expressions.
<br> 
The goal of EmotionLens is to provide a platform (mobile application) for the instructors at the Universities that provide statistics for their students emotions during lectures.

## Installation
To install the project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/baselhusam/EmotionLens.git
```

2. Navigate to the project directory:
```
cd EmotionLens
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

<br>

## Usage
Detailed instructions on how to use the project will be provided here.

<br> 
To run the main script of our model and running the inference live, write the following command:

```bash
python main.py
```
Please note that the model should be download by the step below (**Model** step).

<br>

**NOTE:** the `rmn.py` script is a test script for a new technique that is implemented using the ResNet module called **ResdiualMaskingNetwork**. The code is under testing for now so it is not finalized yet.
<br> To run the `rmn.py` script you should clone the official repo of it from the following [link](https://github.com/phamquiluan/ResidualMaskingNetwork) using the following command:

```
git clone git@github.com:phamquiluan/ResidualMaskingNetwork.git
cd ResidualMaskingNetwork
pip install -e .
cd ..
python rmn.py
```

<br>

## Model
Our trained model is not included in this repository due to GitHub's file size limit. However, you can download it from our Google Drive here. The link directs you to an assets folder which contains the trained model. [The Link is here](https://drive.google.com/drive/folders/1xhvKj0xN5g3y93gMM239gCV7YlRiKqsU?usp=sharing) 
<br>

## Contributers
This project has been done by the University of Jordan students:
- Basel Husam
- Jumana Dyab
- Dina Khader
- Shahid Sobani


## Note
This repo containt just the prototype of the project code base, it is not finished yet. Also, it requires many modification in it, from model structure, to the inference part. With your notes and supervising, we believe that we will provide the best code structure for this project.
