import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

# Load data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/preprocessed_data.csv")

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/05_models/gradient_boosting_model.pkl")
print("Gradient Boosting model saved to models/gradient_boosting_model.pkl")
