import cv2
import mediapipe as mp
import csv

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# The sign you are currently teaching the AI
SIGN_NAME = "Namaste" 

print(f"Ready to record '{SIGN_NAME}'. Press 's' to start saving frames.")

collecting = False
records = []

while len(records) < 300: # We want 300 samples per sign
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # Create a list of 84 zeros (2 hands * 21 points * 2 coordinates [x,y])
        row = [0] * 84 
        
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            # i=0 is first hand, i=1 is second hand
            start_index = i * 42
            for j, lm in enumerate(hand_lms.landmark):
                row[start_index + (j * 2)] = lm.x
                row[start_index + (j * 2) + 1] = lm.y
        
        if collecting:
            row.append(SIGN_NAME)
            records.append(row)
            cv2.putText(frame, f"Recording: {len(records)}", (10, 50), 1, 2, (0, 0, 255), 2)

    cv2.imshow("ISL Data Collector", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): collecting = True
    if key == ord('q'): break

# Save to CSV
with open("isl_dataset.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(records)

cap.release()
cv2.destroyAllWindows()