import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load your collected ISL data
df = pd.read_csv('isl_dataset.csv', header=None)

# 2. Split data into 'Features' (X) and 'Labels' (y)
X = df.iloc[:, :-1] # All the coordinates
y = df.iloc[:, -1]  # The sign names (e.g., 'A', 'B', 'Namaste')

# 3. Train/Test split (80% for learning, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 4. Initialize the Random Forest Classifier
# This is great for high-dimensional data like 84 hand points
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Check how smart the AI is
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {score * 100:.2f}%")

# 6. Save the 'Brain' so we can use it in our webcam script
with open('isl_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)