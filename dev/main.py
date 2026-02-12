import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback
)

detector = HandLandmarker.create_from_options(options)

model = None
try:
    if os.path.exists('./isl_model.p'):
        model_dict = pickle.load(open('./isl_model.p', 'rb'))
        model = model_dict['model']
        print("Model loaded successfully!")
    else:
        print("Model file not found. Running in detection mode only.")
except Exception as e:
    print(f"Error loading model: {e}")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("ISL Interpreter Started. Press 'q' to exit.")

def extract_hand_features(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y])
    return coords


def normalize_coordinates(coords):
    if len(coords) < 4:
        return coords

    wrist_x, wrist_y = coords[0], coords[1]
    normalized = []

    for i in range(0, len(coords), 2):
        normalized.extend([
            coords[i] - wrist_x,
            coords[i + 1] - wrist_y
        ])

    return normalized

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)

    if latest_result and latest_result.hand_landmarks:

        all_hand_coords = []

        for i, hand_landmarks in enumerate(latest_result.hand_landmarks):

            # Draw landmarks manually
            for lm in hand_landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Extract & normalize
            hand_coords = extract_hand_features(hand_landmarks)
            normalized_coords = normalize_coordinates(hand_coords)
            all_hand_coords.extend(normalized_coords)

            # Get Left / Right label
            if latest_result.handedness:
                hand_label = latest_result.handedness[i][0].category_name
                wrist_x = int(hand_landmarks[0].x * width)
                wrist_y = int(hand_landmarks[0].y * height)

                cv2.putText(image, f"{hand_label} Hand",
                            (wrist_x, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

        if model is not None and len(all_hand_coords) > 0:
            try:
                feature_size = 84

                if len(all_hand_coords) < feature_size:
                    all_hand_coords.extend([0] * (feature_size - len(all_hand_coords)))
                elif len(all_hand_coords) > feature_size:
                    all_hand_coords = all_hand_coords[:feature_size]

                prediction = model.predict([all_hand_coords])
                predicted_character = str(prediction[0])
                confidence = model.predict_proba([all_hand_coords]).max()

                prediction_text = f"Sign: {predicted_character} ({confidence:.2f})"

                cv2.putText(image, prediction_text,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            except Exception as e:
                cv2.putText(image, "Prediction Error",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
                print(f"Prediction error: {e}")

        else:
            num_hands = len(latest_result.hand_landmarks)
            status_text = f"{num_hands} Hand(s) Detected"
            if model is None:
                status_text += " (No Model Loaded)"

            cv2.putText(image, status_text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    else:
        cv2.putText(image, "No Hands Detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow('Indian Sign Language Interpreter', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("ISL Interpreter stopped.")
