# app.py


import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load model and columns (relative path)
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


st.title("🏦 Loan Approval Predictor")


# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term (days)", [360, 180, 240, 300])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])


# Prepare input dictionary
input_dict = {
"ApplicantIncome": applicant_income,
"CoapplicantIncome": coapplicant_income,
"LoanAmount": loan_amount,
"Loan_Amount_Term": loan_term,
"Credit_History": credit_history,
f"Gender_{gender}": 1,
f"Married_{married}": 1,
f"Education_{education}": 1,
f"Self_Employed_{self_employed}": 1,
f"Property_Area_{property_area}": 1
}


# Predict
if st.button("Predict"):
    # Convert to DataFrame
    input_data = pd.DataFrame([input_dict])
    print(input_data.iloc[:,:5])
    
    # Add missing columns with 0
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match model
    input_data = input_data[model_columns]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    st.dataframe(input_data)
    st.write("Raw Prediction:",prediction)
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader(f"Loan Application Result: {result}")
