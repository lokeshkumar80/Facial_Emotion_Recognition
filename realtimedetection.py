import cv2
import numpy as np
from tensorflow.keras.models import model_from_json




# =====================================================
# 1. CHOOSE WHICH MODEL TO USE (UNCOMMENT EXACTLY ONE)
# =====================================================

# ACTIVE_MODEL = "BASE"        # uses facialemotionmodel.json + facialemotionmodel.weights.h5
# ACTIVE_MODEL = "INITIAL"     # uses initialfacialemotionmodel.json + initialfacialemotionmodel.weights.h5
ACTIVE_MODEL = "FINAL"         # uses finalfacialemotionmodel.json + finalfacialemotionmodel.weights.h5


# =====================================================
# 2. CONFIG FOR ALL MODELS (COMMON SETTINGS)
# =====================================================

# All models are trained with input shape (48, 48, 1) and grayscale
IMG_SIZE = (48, 48)

# Assuming same 7-class order in all three trainings.
# Change labels/order ONLY if you changed it in training.
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise',
}

# Haar cascade for face detection
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)


# =====================================================
# 3. MODEL LOADER
# =====================================================

def load_selected_model():
    model_files = {
        "BASE": {
            "json": "facialemotionmodel.json",
            "weights": "facialemotionmodel.weights.h5",
        },
        "INITIAL": {
            "json": "initialfacialemotionmodel.json",
            "weights": "initialfacialemotionmodel.weights.h5",
        },
        "FINAL": {
            "json": "finalfacialemotionmodel.json",
            "weights": "finalfacialemotionmodel.weights.h5",
        },
    }

    if ACTIVE_MODEL not in model_files:
        raise ValueError(
            f"Unknown ACTIVE_MODEL '{ACTIVE_MODEL}'. Use 'BASE', 'INITIAL', or 'FINAL'."
        )

    json_path = model_files[ACTIVE_MODEL]["json"]
    weights_path = model_files[ACTIVE_MODEL]["weights"]

    print(f"[INFO] Loading {ACTIVE_MODEL} model...")
    print(f"       JSON:    {json_path}")
    print(f"       WEIGHTS: {weights_path}")

    # Load JSON architecture
    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # Load the weights
    model.load_weights(weights_path)

    print("[INFO] Model loaded successfully ✔")

    return model



# =====================================================
# 4. PREPROCESSING
# =====================================================

def preprocess_face(face_gray):
    """
    face_gray: cropped face region in grayscale (H, W)
    Returns image ready for model.predict(): shape (1, 48, 48, 1)
    """
    # Resize to match training input size
    face_resized = cv2.resize(face_gray, IMG_SIZE)

    # Normalize to [0, 1]
    face_resized = face_resized.astype("float32") / 255.0

    # (H, W) -> (H, W, 1)
    face_resized = np.expand_dims(face_resized, axis=-1)

    # Add batch dimension: (1, H, W, 1)
    face_resized = np.expand_dims(face_resized, axis=0)

    return face_resized


# =====================================================
# 5. REAL-TIME LOOP
# =====================================================

def main():
    model = load_selected_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"[INFO] Real-time detection started using model: {ACTIVE_MODEL}")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Convert frame to grayscale for detection and model input
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces on grayscale image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around face on the original BGR frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face region from grayscale image
            face_roi_gray = gray[y:y + h, x:x + w]

            # Preprocess for the model
            face_input = preprocess_face(face_roi_gray)

            # Predict emotion
            preds = model.predict(face_input, verbose=0)
            emotion_idx = int(np.argmax(preds))
            emotion_label = EMOTION_LABELS.get(emotion_idx, "unknown")
            confidence = float(np.max(preds))

            # Put label and confidence above the face
            text = f"{emotion_label} ({confidence:.2f})"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Show the frame
        cv2.imshow("Real-time Facial Emotion Recognition", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





# To run the file
#  python3 realtimedetection.py
