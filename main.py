import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- 1. INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Hands module
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load trained model with error handling
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

# Initialize camera with error handling
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("ISL Interpreter Started. Press 'q' to exit.")

def extract_hand_features(hand_landmarks):
    """Extract normalized coordinates from hand landmarks"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y])
    return coords

def normalize_coordinates(coords):
    """Normalize coordinates relative to wrist position"""
    if len(coords) < 4:
        return coords
    
    # Use wrist (first landmark) as reference point
    wrist_x, wrist_y = coords[0], coords[1]
    normalized = []
    
    for i in range(0, len(coords), 2):
        normalized.extend([
            coords[i] - wrist_x,      # Relative x
            coords[i + 1] - wrist_y   # Relative y
        ])
    
    return normalized

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for selfie-view
    image = cv2.flip(image, 1)
    height, width, _ = image.shape
    
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # --- 2. SIGN RECOGNITION LOGIC ---
    if results.multi_hand_landmarks:
        all_hand_coords = []
        
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract and normalize coordinates
            hand_coords = extract_hand_features(hand_landmarks)
            normalized_coords = normalize_coordinates(hand_coords)
            all_hand_coords.extend(normalized_coords)
            
            # Get hand label (Left/Right)
            hand_label = results.multi_handedness[i].classification[0].label
            
            # Draw hand label
            cv2.putText(image, f"{hand_label} Hand", 
                       (int(hand_landmarks.landmark[0].x * width), 
                        int(hand_landmarks.landmark[0].y * height) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # --- 3. PREDICTION ---
        if model is not None and len(all_hand_coords) > 0:
            try:
                # Ensure consistent input size (pad or truncate as needed)
                feature_size = 84  # Adjust based on your model's expected input
                if len(all_hand_coords) < feature_size:
                    all_hand_coords.extend([0] * (feature_size - len(all_hand_coords)))
                elif len(all_hand_coords) > feature_size:
                    all_hand_coords = all_hand_coords[:feature_size]
                
                # Make prediction
                prediction = model.predict([all_hand_coords])
                predicted_character = str(prediction[0])
                confidence = model.predict_proba([all_hand_coords]).max()
                
                # Display prediction
                prediction_text = f"Sign: {predicted_character} ({confidence:.2f})"
                cv2.putText(image, prediction_text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            except Exception as e:
                cv2.putText(image, f"Prediction Error", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Prediction error: {e}")
        else:
            # Show detection status
            num_hands = len(results.multi_hand_landmarks)
            status_text = f"{num_hands} Hand(s) Detected"
            if model is None:
                status_text += " (No Model Loaded)"
            cv2.putText(image, status_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # No hands detected
        cv2.putText(image, "No Hands Detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 4. DISPLAY ---
    cv2.imshow('Indian Sign Language Interpreter', image)

    # Controls
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("ISL Interpreter stopped.")
