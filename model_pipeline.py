import sys
import os
import io
import json
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure standard output can print unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Starting Data Analysis and Model Pipeline...")

# Load Data
df = pd.read_excel('Data.xlsx', sheet_name='Synthetic Responses')
print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

# Clean Likert Scale Questions
# Q5: AI Usefulness (1 to 5)
q5_map = {'Not at all useful': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Extremely useful': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q5_num'] = df['Q5_AI_Usefulness'].map(q5_map).astype(float)

# Q6: Free Tool Limitations (1 to 5)
q6_map = {'Never': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Very frequently': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q6_num'] = df['Q6_Free_Tool_Limitations'].map(q6_map).astype(float)

# Q9: AI Dependence (1 to 5)
q9_map = {'Not dependent at all': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Extremely dependent': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q9_num'] = df['Q9_AI_Dependence'].map(q9_map).astype(float)

# Q27: Likelihood to Pay in Next 12 Months (1 to 5)
q27_map = {'Very Unlikely': 1, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'Very Likely': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
df['Q27_num'] = df['Q27_Likelihood_to_Pay_Next_12_Months'].map(q27_map).astype(float)

# Q3 Frequency ordinal mapping (estimated daily uses or scale 1 to 5)
q3_map = {
    'Never': 0,
    'Rarely': 1,
    'A few times a month': 2,
    'A few times a week': 3,
    'Once a day': 4,
    'Multiple times a day': 5
}
df['Q3_num'] = df['Q3_AI_Use_Frequency'].map(q3_map).astype(float)

# Q8 Daily AI Time ordinal mapping (hours/day estimate)
q8_map = {
    'I rarely/never use AI': 0.0,
    'Less than 15 minutes': 0.15,
    '15-30 minutes': 0.375,
    '30-60 minutes': 0.75,
    '1-2 hours': 1.5,
    'More than 2 hours': 2.5
}
df['Q8_hours'] = df['Q8_Daily_AI_Time'].map(q8_map).astype(float)

# Q11 Free Tool Failure frequency (1 to 5)
q11_map = {
    'Never': 1,
    'Rarely': 2,
    'Sometimes': 3,
    'Often': 4,
    'Very often': 5
}
df['Q11_num'] = df['Q11_Free_Tool_Failure'].map(q11_map).astype(float)

# Q18 Max Willingness to Pay numerical midpoint
# Brackets: I would not pay (0), Less than 100 (50), 100-199 (150), 200-299 (250), 300-399 (350), 400-499 (450), 500-699 (600), 700 or more (800)
q18_midpoint = {
    'I would not pay': 0,
    'Less than ₹100': 50,
    '₹100–₹199': 150,
    '₹200–₹299': 250,
    '₹300–₹399': 350,
    '₹400–₹499': 450,
    '₹500–₹699': 600,
    '₹700 or more': 800
}
df['WTP_midpoint'] = df['Q18_Max_Willingness_to_Pay'].map(q18_midpoint).astype(float)

# Q16 Monthly Value numerical midpoint
q16_midpoint = {
    'I cannot estimate': np.nan,
    'Less than ₹100': 50,
    '₹100–₹249': 175,
    '₹250–₹499': 375,
    '₹500–₹749': 625,
    '₹750–₹999': 875,
    '₹1,000 or more': 1200
}
df['Perceived_Value'] = df['Q16_Monthly_Value'].map(q16_midpoint)

# Branch classification
def classify_branch(branch):
    if any(k in branch for k in ['Computer Science', 'Artificial Intelligence', 'Information Science']):
        return 'CS_AI_IT'
    elif any(k in branch for k in ['Electronics', 'Electrical']):
        return 'ECE_EEE'
    else:
        return 'Core_Other'

df['Branch_Category'] = df['Q2_Engineering_Branch'].apply(classify_branch)

# Year numerical
df['Year_Num'] = df['Q1_Year_of_Study'].str.extract('(\d+)').astype(float)

# Current Paying status binary
df['Is_Current_Payer'] = df['Q14_Current_Payment_Status'].apply(lambda x: 1 if 'Yes' in str(x) else 0)

# High Intent to Pay Binary (Q27 >= 4 or Current Payer)
df['High_Pay_Intent'] = ((df['Q27_num'] >= 4) | (df['Is_Current_Payer'] == 1)).astype(int)

print("Data Cleaning Complete. Sample summary statistics:")
print(df[['Q5_num', 'Q6_num', 'Q9_num', 'Q11_num', 'Q27_num', 'WTP_midpoint', 'High_Pay_Intent']].describe())
