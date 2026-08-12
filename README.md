Overview

EcoCycle is an AI-powered web application that promotes sustainable item management by classifying old or used items into three categories: Reuse, Repair, and Recycle. The system utilizes MobileNetV2, OpenCV, and Flask to analyze uploaded images and provide sustainability-oriented recommendations.

Features
Image-based classification of old items
Reuse, Repair, and Recycle categorization
Multi-object detection using OpenCV
MobileNetV2 transfer learning model
User-friendly web interface
Sustainable decision support for resource management
Technology Stack
Python
Flask
TensorFlow / Keras
MobileNetV2
OpenCV
HTML
CSS
JavaScript
NumPy
Dataset

A custom dataset was created containing images organized into three categories:

Reuse
Repair
Recycle

Data augmentation techniques were applied to improve model performance and generalization.

Working
User uploads an image of an old item.
OpenCV preprocesses the image and detects object regions.
Each detected object is processed individually.
MobileNetV2 extracts features and performs classification.
The system predicts whether the item should be Reused, Repaired, or Recycled.
Results are displayed through the web interface.


Project Objective

To develop an intelligent platform that assists users in making sustainable decisions regarding old items by leveraging Artificial Intelligence and Computer Vision technologies.

Future Enhancements
Larger and more diverse dataset
Cloud deployment
Mobile application support
Real-time video analysis
Enhanced marketplace integration
