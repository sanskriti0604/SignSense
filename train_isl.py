import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. Load your combined Kaggle data
data = pd.read_csv('combined_isl_data.csv')
X = data.iloc[:, :-1] # Landmarks
y = data.iloc[:, -1]  # Labels (A, B, C, Namaste, etc.)

# 2. Split for validation
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# 4. Save the model to your 'models/' folder
with open('isl_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Training complete! Model saved as isl_model.p")