# Facial Emotion Recognition 😄😠😢

A deep learning-based system to detect human emotions from facial expressions using CNNs, trained on the FER-2013 dataset.

<!-- optional, you can upload a banner -->

---

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to recognize facial expressions in real-time or from static images. It supports 7 key emotions:

- 😄 Happy
- 😠 Angry
- 😢 Sad
- 😮 Surprise
- 😐 Neutral
- 😨 Fear
- 🤢 Disgust

---

## 🚀 Features

- Real-time facial emotion detection using OpenCV
- Model trained on the FER-2013 dataset
- Simple and clean GUI for image-based emotion detection
- CNN with high accuracy on validation/test sets
- Easy to extend and integrate into other applications

---

## 🧠 Model Architecture

The CNN is built using the following structure:

- 3 Convolutional layers
- 2 MaxPooling layers
- Dropout layers for regularization
- Fully connected Dense layers
- Output layer with softmax activation for 7 classes

Input -> Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Conv2D -> Flatten -> Dense -> Output

**FER-2013 (Facial Expression Recognition 2013)**  
- Source: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- 35,887 grayscale images (48x48)
- 7 emotions labeled
- Split: 28,709 training / 3,589 validation / 3,589 test

---

## 🛠️ Setup & Installation

### 📦 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt

// Or manually:

pip install tensorflow keras opencv-python matplotlib numpy pandas

```
### ▶️ How to Run

🧪 Training the Model

```bash
python trainmodel.py

```
This trains the CNN model on the FER-2013 dataset.

### 🎯 Testing the Model

```bash
python emotiondetector.py

```
This runs the model on live webcam feed or test images.



