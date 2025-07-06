# train_model.py


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


# Load dataset
df = pd.read_csv("C:/Users/C8C7KG/Djangolearnings/loan_approval_system/train_u6lujuX_CVtuZ9i.csv")


# Drop Loan_ID and handle missing
df = df.drop("Loan_ID", axis=1)
df.fillna(method="ffill", inplace=True)


# Convert categorical to numeric
df = pd.get_dummies(df)


# Pick 'Loan_Status_Y' if it exists, otherwise convert manually
if 'Loan_Status_Y' in df.columns:
    y = df['Loan_Status_Y']
    X = df.drop(['Loan_Status_Y', 'Loan_Status_N'], axis=1)
else:
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    X = df.drop('Loan_Status', axis=1)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Save model
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)


# Save column names (for input form)
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)