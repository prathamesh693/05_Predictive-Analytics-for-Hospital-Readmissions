import pandas as pd
import os

# Load the preprocessed data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/preprocessed_data.csv")

# Example 1: Age group bucketing
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle-aged', 'senior', 'elderly'])

# Example 2: Creating a feature for total visits
visit_columns = ['inpatient_visits', 'outpatient_visits', 'emergency_visits']
if all(col in df.columns for col in visit_columns):
    df['total_visits'] = df['inpatient_visits'] + df['outpatient_visits'] + df['emergency_visits']

# Example 3: Binary flag for chronic condition
if 'num_diagnoses' in df.columns:
    df['many_diagnoses'] = (df['num_diagnoses'] >= 5).astype(int)

# Convert any new categorical features
if 'age_group' in df.columns:
    df['age_group'] = df['age_group'].astype(str).astype('category').cat.codes

# Save engineered dataset
df.to_csv("R:/Projects/1_Data_Science & ML_Projects/05_Predictive Analytics for Hospital Readmissions/02_Dataset/featured-engineered_data.csv", index=False)
print(f"Feature-engineered data saved")