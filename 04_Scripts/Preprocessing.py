import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the raw dataset
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/hospital_readmissions.csv")

# Drop identifier columns
for col in ['encounter_id', 'patient_nbr']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Encode target variable correctly
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# Separate features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from numeric columns before scaling
numeric_cols = [col for col in numeric_cols if col != 'readmitted']

# Label encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Impute and scale
imputer = SimpleImputer(strategy='median')
df[numeric_cols + categorical_cols] = imputer.fit_transform(df[numeric_cols + categorical_cols])

# Scale numeric features (excluding target)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save processed data
output_path = "R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/preprocessed_data.csv"
df.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
