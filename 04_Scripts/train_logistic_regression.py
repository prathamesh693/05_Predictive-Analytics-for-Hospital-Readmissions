import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
# Load processed data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/preprocessed_data.csv")

# Separate features and target
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/05_models/logistic_regression_model.pkl")
print("Logistic Regression model saved to models/logistic_regression_model.pkl")
