# Facial Emotion Recognition (Deep Learning – CNN)

This repository implements a complete **Facial Emotion Recognition (FER)** system using deep learning (TensorFlow/Keras).  
It includes multiple trained models, evaluation metrics, confusion matrices, and a real-time emotion detection system using OpenCV.

The system classifies facial expressions into the following emotions:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## 🚀 Key Features

- Multiple training experiments (Initial, Intermediate, Final)
- JSON + Keras model + weights for reproducibility
- Real-time emotion detection via webcam
- Confusion matrices (CSV + PNG)
- Classification reports
- Accuracy & loss curve visualizations
- Reproducible environment (requirements.txt + environment.yml)

---

## 📂 Project Structure
-├──Facial_Emotion_Recognition # Project folder

-├── FINAL_TRAINING.ipynb # Final training experiment

-├── INITIAL_TRAINING.ipynb # Initial training experiment

-├── TRAINING.ipynb # Intermediate training
-|
-├── realtimedetection.py # Real-time emotion detection
-├── oldrealtimedetection.py # Older version of detector
-│
-├── facialemotionmodel.json # Intermediate model architecture
-├── facialemotionmodel.keras # Intermediate model
-├── facialemotionmodel.weights.h5 # Intermediate weights
-│
-├── initialfacialemotionmodel.json # Initial model architecture
-├── initialfacialemotionmodel.keras # Initial model
-├── initialfacialemotionmodel.weights.h5 # Initial weights
-│
-├── finalfacialemotionmodel.json # Final model architecture
-├── finalfacialemotionmodel.keras # Final trained model (recommended)
-├── finalfacialemotionmodel.weights.h5 # Final weights
-│
-├── best_emotion_cnn.weights.h5 # Best-performing model weights
-│
-├── Confusion_matrix_FINAL_TRAINING.png
-├── Confusion_matrix_INITIAL_TRAINING.png
-├── Confusion_matrix_TRAINING.png
-│
-├── confusion_matrix_FINAL_TRAINING.csv
-├── confusion_matrix_INITIAL_TRAINING.csv
-├── confusion_matrix_TRAINING.csv
-│
-├── accuracy_curve_final.png
-├── accuracy_curve_initial.png
-├── accuracy_curve_intermediate.png
-│
-├── loss_curve_final.png
-├── loss_curve_initial.png
-├── loss_curve_intermediate.png
-│
-├── classification_report_FINAL_TRAINING.txt
-├── classification_report_INITIAL_TRAINING.txt
-├── classification_report_TRAINING.txt
-│
-├── images/
-│ ├── train/ # Training dataset
-│ └── test/ # Testing dataset
-│ ├── angry/
-│ ├── disgust/
-│ ├── fear/
-│ ├── happy/
-│ ├── neutral/
-│ ├── sad/
-│ ├── surprise/
-│
-├── requirements.txt # Python dependencies
-├── environment.yml # Conda environment
-└── README.md

---

## 🧠 Model Overview

CNN-based architecture used in three stages:

### **Initial Model**
- Baseline performance  
- Fewer epochs  

### **Intermediate Model**
- Improved tuning & accuracy  

### **Final Model (Recommended)**
- Best accuracy  
- Use:
  - `finalfacialemotionmodel.json`
  - `finalfacialemotionmodel.keras`
  - `finalfacialemotionmodel.weights.h5`

---

## ⚙️ Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/lokeshkumar80/Facial_Emotion_Recognition.git

cd Facial_Emotion_Recognition

2️⃣ Install Dependencies

Option A: Using pip
bash
pip install -r requirements.txt

Option B: Using Conda
bash
conda env create -f environment.yml
conda activate facial_emotion_env
Includes TensorFlow + NumPy versions that avoid compatibility issues.

🧪 Training the Model
Use any of the training notebooks:

INITIAL_TRAINING.ipynb

TRAINING.ipynb

FINAL_TRAINING.ipynb

Each notebook includes preprocessing, model creation, training, saving weights, and visualizations.

📊 Evaluation Results
Included in the repo:

Confusion matrices (PNG + CSV)

Classification reports (TXT)

Accuracy curves

Loss curves

These help compare performance across training stages.

🎥 Real-Time Emotion Detection
To run the live webcam detector:

bash
python realtimedetection.py

This script:
Loads the final trained model

Detects faces via OpenCV

Predicts emotion

Displays real-time results

Press Q to exit.

🛠️ Technologies Used
Python 3.10+

TensorFlow / Keras

NumPy, Pandas

OpenCV

Matplotlib

Jupyter Notebook

📌 Troubleshooting
❗ NumPy 1.x vs 2.x TensorFlow Error
Solution:

bash
pip install -r requirements.txt

❗ Model Load Error
python
from tensorflow.keras.models import model_from_json

with open("finalfacialemotionmodel.json") as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights("finalfacialemotionmodel.weights.h5")

📬 Contact
For improvements or issues, open an Issue or Pull Request on GitHub.
---

If you want:

✅ Badges (TensorFlow, Python version, GitHub stars)  
✅ A project logo  
✅ A screenshot section  
✅ A demo video/GIF section  

Just tell me — I can add them.
