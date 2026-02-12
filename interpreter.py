import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained brain
model_dict = pickle.load(open('isl_model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        data_aux = [0] * 84 # Buffer for 2 hands
        
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            start_idx = i * 42
            for j, lm in enumerate(hand_lms.landmark):
                data_aux[start_idx + (j*2)] = lm.x
                data_aux[start_idx + (j*2) + 1] = lm.y

        # Predict the sign
        prediction = model.predict([data_aux])
        predicted_character = prediction[0]

        cv2.putText(frame, predicted_character, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow('ISL Interpreter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()