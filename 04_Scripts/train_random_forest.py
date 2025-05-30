# - - - train_random_forest.py - - - #
# src/models/train_random_forest.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/preprocessed_data.csv")

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/05_models/random_forest_model.pkl")
print("Random Forest model saved to models/random_forest_model.pkl")
