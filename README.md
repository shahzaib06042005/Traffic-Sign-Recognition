# 🚦 Traffic Sign Recognition

## 📌 Project Overview
This project aims to classify traffic signs into their respective categories using deep learning techniques.
Leveraging Convolutional Neural Networks (CNNs), the model is trained on the GTSRB dataset to identify various traffic sign classes from images.
The goal is to build a robust computer vision system for multi-class classification in real-world scenarios.

## 📂 Dataset
Source: GTSRB - German Traffic Sign Recognition Benchmark (Kaggle)

Description: Contains thousands of labeled images of different traffic sign types.

Target Variable: Traffic sign class (multi-class labels)

## 🛠 Tools & Libraries
Python

TensorFlow / Keras

OpenCV

NumPy, Pandas

Matplotlib, Seaborn (for visualization)

## 🔍 Approach
Data Preprocessing

Loaded and explored dataset images.

Resized all images to 32x32 pixels.

Normalized pixel values to improve model convergence.

Model Building

Designed a CNN architecture with convolution, pooling, and dense layers.

Used ReLU activation for hidden layers and softmax for output layer.

Model Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Early stopping to prevent overfitting.

Model Evaluation

Evaluated performance using accuracy and confusion matrix.

Analyzed misclassifications for improvement opportunities.

## 📊 Results
Achieved high accuracy in traffic sign classification.

Most classes were predicted correctly with minimal misclassification.

## 🚀 How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/shahzaib06042005/traffic-sign-recognition.git
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the model:

bash
Copy
Edit
python traffic_sign_recognition.py
