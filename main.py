import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- 1. INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Hands module (Tracking 2 hands for ISL)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load your trained model (Uncomment these lines after you run train_isl.py)
# model_dict = pickle.load(open('./isl_model.p', 'rb'))
# model = model_dict['model']

cap = cv2.VideoCapture(0)

print("ISL Interpreter Started. Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # --- 2. SIGN RECOGNITION LOGIC ---
    if results.multi_hand_landmarks:
        # Create a buffer for 84 coordinates (21 pts * 2 coords * 2 hands)
        data_aux = [0] * 84 
        
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw the hand landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract (x, y) coordinates for this hand
            start_idx = i * 42
            for j, lm in enumerate(hand_landmarks.landmark):
                data_aux[start_idx + (j * 2)] = lm.x
                data_aux[start_idx + (j * 2) + 1] = lm.y

        # --- 3. PREDICTION PLACEHOLDER ---
        # Once your model is trained, the lines below will show the translated text
        # prediction = model.predict([data_aux])
        # predicted_character = str(prediction[0])
        
        # Temporary status text
        status_text = "Hand(s) Detected"
        cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 4. DISPLAY ---
    cv2.imshow('Indian Sign Language Interpreter', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()