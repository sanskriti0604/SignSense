import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- 1. SETUP & MODEL LOADING ---
# Load the trained 'brain' created by train_isl.py
try:
    model_dict = pickle.load(open('./isl_model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'isl_model.p' not found. Please run train_isl.py first.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, # Critical for two-handed ISL signs
    min_detection_confidence=0.7
)

# --- 2. SENTENCE BUILDING VARIABLES ---
current_sentence = ""
last_predicted = ""
consistency_counter = 0
STABILITY_THRESHOLD = 25  # Number of frames to hold a sign to "type" it

cap = cv2.VideoCapture(0)

print("Interpreter Running. Press 'q' to quit. Press 'c' to clear sentence.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    display_text = "Waiting for signs..."

    if results.multi_hand_landmarks:
        # 84 coordinates (21 pts * 2 coords * 2 hands)
        data_aux = [0] * 84 
        
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2: break 
            
            # Draw the hand skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks for prediction
            start_idx = i * 42
            for j, lm in enumerate(hand_landmarks.landmark):
                data_aux[start_idx + (j * 2)] = lm.x
                data_aux[start_idx + (j * 2) + 1] = lm.y

        # Predict using the model trained from your 3 Kaggle datasets
        prediction = model.predict([data_aux])
        predicted_character = str(prediction[0])
        display_text = f"Sign Detected: {predicted_character}"

        # --- 3. STABILITY & SENTENCE LOGIC ---
        if predicted_character == last_predicted:
            consistency_counter += 1
        else:
            consistency_counter = 0
            last_predicted = predicted_character

        # If the sign is held steady, append to sentence
        if consistency_counter == STABILITY_THRESHOLD:
            if predicted_character == "SPACE": # Handle space signs if in your dataset
                current_sentence += " "
            else:
                current_sentence += predicted_character
            print(f"Typed: {predicted_character}")

    # --- 4. USER INTERFACE (UI) ---
    # Background box for clarity
    cv2.rectangle(frame, (0, 0), (W, 80), (0, 0, 0), -1)
    
    # Show current live prediction
    cv2.putText(frame, display_text, (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the accumulating sentence (The "Interpreter" part)
    cv2.putText(frame, f"Sentence: {current_sentence}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('SignSense ISL Interpreter', frame)

    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): current_sentence = "" # Clear sentence

cap.release()
cv2.destroyAllWindows()