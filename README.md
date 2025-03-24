# Violence-Detection
ABSTRACT

VIOLENCE DETECTION THROUGH OPEN CV


This project presents a comprehensive system for real-time violence detection using human
pose estimation and deep learning techniques. By leveraging the power of MediaPipe for
pose landmark detection and Long Short-Term Memory (LSTM) networks for sequence
classification, we have developed a robust solution capable of identifying violent actions in
real-time video streams. The system is designed to process video input from a camera, extract
relevant pose features using Media Pipe’s pose model, and classify the observed sequences
as either violent or neutral using a trained LSTM model. The project encompasses data
collection of various violent and non-violent actions, model training, and the development
of a real-time detection system with an intuitive user interface. Key features of the system
include multi-threaded processing for improved performance, adaptive video frame cropping
for better focus on the subject, and a flexible architecture that allows for easy expansion to
recognize additional action types. 


Objective:
The primary objective of this project is to develop a robust and efficient system for real-time
violence detection using pose estimation and deep learning techniques. This system aims to provide
accurate and responsive detection and classification of violent actions from live video input.
Specific objectives include:
1. Implement real-time pose estimation using MediaPipe's pose detection model.
2. Develop a violence detection system using pose landmark data.
3. Train an LSTM model for sequence-based classification of violent and non-violent actions.
4. Create an efficient multi-threaded architecture for real-time processing.
5. Design an intuitive user interface for visualization and interaction.
Aim:

The aim of this project is to advance the field of automated surveillance and public safety by
providing a versatile tool for detecting and alerting violent behavior. By achieving accurate real-time
detection and classification, this system aims to enable a wide range of applications, including:
1. Enhancing security in public spaces such as schools, malls, and transportation hubs.
2. Improving response times for law enforcement and security personnel.
3. Providing early warning systems for potential violent incidents.
4. Assisting in post-incident analysis and investigation.
5. Supporting research in behavioral analysis and violence prevention.
Through the development of this system, we aim to demonstrate the potential of combining
stateofthe-art computer vision techniques with deep learning models to create practical and
responsive violence detection interfaces.


 2
 SYSTEM ANALYSIS

2.1 EXISTING SYSTEM:

● Anchal Sood and Anju Mishra have proposed a sign recognition system based on
Harris algorithm for extraction of feature in which after the image pre-processing
part, the feature is extracted and stored in the Nx2 matrix. This matrix is further used
to match the image from the database. There are some limitations to the system. The
very light brown to somewhat dark brown background gives error as they are
considered in the range value for skin segmentation. But the results are efficient.
● Prashant G. Ahire, Kshitija B. Tilekar, Tejaswini A. Jawake, Pramod B. Warale
system works on MATLAB for Sign Language recognition, here they have taken a
real time video as an input and after applying the image processing stage they have
used the correlation based approach for mapping purpose and then at last the audio is
generated using google TTS API.
3.1.1 DISADVANTAGES OF EXISTING SYSTEM:

● It difficult to understand what exactly the other person is trying to express and so with
the deaf people. Sometimes people interpret these messages wrongly either through
sign language or through lip reading or lip sync.
2.2 PROPOSED SYSTEM:
● In this paper author is building machine learning model which can predict Sign
Language from webcam and then convert recognize gesture into voice so normal
peoples can understand what Deaf and Dumb peoples are saying.
● In propose paper author has used SVM algorithm but in python SVM is not accurate
in identifying Sign Language so we are using deep learning Convolution Neural
Network to train Sign Language images and then this trained model can be used to
predict those trained Sign Language from webcam.
3.2.1 ADVANTAGES OF PROPOSED SYSTEM:

● Classification of an images is done through CNN and SVM algorithm.
● Implementation of this system gives up to 90% accuracy and works successfully in
most of the test cases.

 3
2.3 SYSTEM REQUIREMENTS:
To ensure optimal performance of the Real-time Violence Detection System, the following hardware
and software requirements should be met:
 1. Hardware Requirements: a. Processor: o Minimum: Intel Core i5
(8th generation) or AMD Ryzen 5 (3000 series) o Recommended: Intel Core i7
 (10th generation) or AMD Ryzen 7 (5000 series) b. RAM: o Minimum: 8 GB o
Recommended: 16 GB or higher
c. Graphics Card:
o Minimum: NVIDIA GeForce GTX 1050 or AMD Radeon RX 560 o Recommended:
NVIDIA GeForce RTX 2060 or AMD
Radeon RX 5700 d. Storage:
o Minimum: 256 GB SSD o Recommended: 512 GB
SSD or higher
e. Camera:
o Minimum: 720p webcam with 30 fps o Recommended:
1080p webcam with 60 fps
2. Software Requirements: a. Operating System: o
Windows 10 (64-bit) or later o Ubuntu 20.04 LTS
or later o macOS 10.15 (Catalina) or
later
b. Python:
o Version 3.8 or later

c. Required Python Libraries:
o OpenCV (cv2): 4.5.0 or
 later o TensorFlow: 2.4.0 or
 4
later o MediaPipe: 0.8.3 or later o
NumPy: 1.19.0 or later o
Pandas: 1.2.0 or later o
h5py: 3.1.0 or later
d. Additional Software: o CUDA Toolkit: 11.0 or later (for NVIDIA GPUs) o cuDNN: 8.0
or later (for NVIDIA GPUs)
3. Network Requirements:
o Stable internet connection for initial setup and potential cloud integration o
Minimum upload speed of 5 Mbps for remote monitoring features
4. Environmental Requirements:
o Well-lit area for optimal camera performance o Sufficient space for subject
movement during action
detection
5. Optional Requirements (for enhanced performance):
 o Multiple cameras for multi-angle detection o High-performance
workstation for handling multiple video streams o Dedicated server for data storage and
processing in large-scale deployments
6. Development Environment: o Integrated Development Environment (IDE):
PyCharm, Visual Studio Code, or Jupyter
 Notebook o Version
Control: Git 2.30 or later
7. Deployment Considerations:
 o Docker 20.10 or later for containerized deployment o Kubernetes 1.20 or
later for orchestrating multiple instances in large-scale setups
Meeting these system requirements will ensure smooth operation of the Real-time Violence
Detection System, enabling accurate and responsive detection of violent actions in various
environments. It's important to note that while the system can run on minimum specifications, the
recommended specifications will provide better performance, especially when processing
highresolution video streams or handling multiple cameras simultaneously.
 Technologies and Libraries Used :
 5
The project leverages several key technologies and libraries to achieve its objectives. This section
provides an overview of the main tools used and their roles in the system.
6.1 OpenCV (cv2)
OpenCV (Open Source Computer Vision Library) is a crucial component of our system, primarily
used for video capture and image processing tasks.
Key uses in the project:
• Video capture from camera input
• Frame rotation and cropping
• Drawing landmarks, connections, and bounding boxes
• Displaying processed frames with overlaid information
Example from the code:
python
import cv2 cap = cv2.VideoCapture(0) ret, frame = cap.read() cv2.imshow("image",
frame)
6.2 MediaPipe
MediaPipe is a cross-platform framework for building multimodal applied machine learning pipelines.
In this project, it's used for pose estimation.
Key uses in the project:
• Detecting pose landmarks
• Providing pre-trained models for efficient real-time detection
Example from the code:

import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose() results =
pose.process(frameRGB)

6.3 TensorFlow
TensorFlow is an open-source machine learning framework. In this project, it's primarily used for
loading and running the trained LSTM model.
Key uses in the project:
• Loading trained LSTM model
 6
• Performing inference for violence classification
Example from the code:

import tensorflow as tf

model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects) result = model.predict(lm_list)
6.4 NumPy and Pandas
NumPy and Pandas are essential libraries for numerical computing and data manipulation in Python.
Key uses in the project:
• Array operations for landmark data
• Data preprocessing and normalization
• Storing and manipulating collected data
Example from the code:

import numpy as np import pandas as pd

lm_list = np.array(lm_list) lm_list =
np.expand_dims(lm_list, axis=0) df =
pd.DataFrame(lm_list) df.to_csv(label +
".txt")
Additional Libraries:
1. threading: Used for implementing multi-threading to improve real-time performance.
2. h5py: Used for working with HDF5 files, particularly for loading the trained models.
3. json: Used for parsing and manipulating JSON data, especially in model loading.
The combination of these technologies and libraries enables the system to perform efficient real-time
violence detection, from video capture to final classification and visualization.
7. Data Collection and Preprocessing
