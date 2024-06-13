# Real Time Object-Detection, Tracking and Counting System 
![Screenshot at 2023-09-16 01-31-17](https://github.com/Tuning-AI/Real-time-Object-Detection-and-Tracking-and-Counting-system/assets/99510125/53c3d067-7a04-45c7-bdd2-8d7220b8f36a)

## Overview
Object detection and tracking and counting in real-time using YOLO V8 and SORT algorithm" is a computer vision project that aims to 
detect and track objects in real-time video streams using YOLO V8, a state-of-the-art object detection algorithm, and the Simple Online and Realtime
Tracking (SORT) algorithm, a popular object tracking algorithm. The project involves preprocessing the video stream, applying object detection to 
detect and classify objects in the scene, and then using SORT algorithm to track the detected objects across subsequent frames in the video. 
The project also includes the development of a counting mechanism to count the number of detected and tracked objects in the video stream.
The application of this project can be found in various domains like traffic monitoring, crowd control, and security surveillance.

## YOLOv8
YOLOv8 is a state-of-the-art object detection and image segmentation model developed by Ultralytics. It is the latest version of the YOLO (You Only Look Once) family of models, which are known for their speed and accuracy.

## SORT
SORT is a method that estimates the location of an object based on its past location using the Kalman filter.
The Kalman filter is quite effective against occlusion. SORT is comprised of three components: Detection: Firstly, the initial object of interest is detected.

## Frameworks

* **Streamlit** is a library for building web apps. It is easy to use and does not require any knowledge of web development.
* **OpenCV-Python** is a library for computer vision. It provides a wide range of functions for image processing, object detection, and machine learning.
* **cvzone** is a library for computer vision. It provides a number of functions for image processing and object detection.
* **Ultralytics** is a library for object detection. It provides a number of pre-trained models for object detection.
* **Scikit-image** is a library for image processing. It provides a wide range of functions for image processing, including segmentation, filtering, and feature extraction.
* **Matplotlib** is a library for plotting data. It provides a number of functions for creating graphs and charts.
* **NumPy** is a library for scientific computing. It provides a high-performance array data type and a number of functions for mathematical operations.
* **Filterpy** is a library for filtering data. It provides a number of functions for filtering temporal data, such as Kalman filters and particle filters.
* **Lap** is a library for image processing. It provides a number of functions for image segmentation and edge detection.

## How To Run This APP
```
pip install -r requirements.txt
streamlit run app.py
```
- Created By Kirouane Ayoub 
