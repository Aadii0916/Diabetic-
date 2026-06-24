import numpy as np
import pandas as pd
import joblib
import streamlit as st
model_dg=joblib.load('dg_model.pkl')
scaler = joblib.load('scaler.pkl')
# try:
#     #model = joblib.load('diabetes_model.pkl')
#     model = joblib.load('dg_model.pkl')
#     scaler = joblib.load('scaler.pkl')
# except FileNotFoundError:
#     st.error("Model or scaler files not found. Please run the `svm_diabetes_classifier.py` script first to create them.")
#     st.stop()

try:
    model = joblib.load('dg_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')  # ← change this
except FileNotFoundError:
    st.error("Model or scaler files not found...")
    st.stop()

# --- WEB APP INTERFACE ---
# Configure the page title and layout
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# Display the main title of the application
st.title('Diabetes Prediction Application')
st.write("This app uses a Machine Learning model to predict whether a person has diabetes based on a few key health metrics.")

# --- USER INPUT SECTION ---
st.header("Enter Patient's Details")

# Create three columns to organize the input fields neatly
col1, col2, col3 = st.columns(3)

# Input fields in the first column
with col1:
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=70)

# Input fields in the second column
with col2:
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

# Input fields in the third column
with col3:
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)

# --- PREDICTION LOGIC ---
# This code runs only when the user clicks the 'Predict Diabetes' button
if st.button('**Predict Diabetes**', key='predict_button'):
    # Step 1: Collect all user inputs into a single list in the correct order
    user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    
    # Step 2: Convert the list of numbers into a NumPy array
    input_data_as_numpy_array = np.asarray(user_input)
    
    # Step 3: Reshape the array because the model expects a 2D array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Step 4: Use the loaded scaler to standardize the user's input data
    std_data = scaler.transform(input_data_reshaped)
    
    # Step 5: Use the trained model to make a prediction on the scaled data
    prediction = model.predict(std_data)
    
    # --- DISPLAY THE RESULT ---
    st.header("Prediction Result")
    # The model outputs 0 for non-diabetic and 1 for diabetic
    if prediction[0] == 0:
        st.success('**The person is NOT Diabetic.**')
    else:
        st.error('**The person IS Diabetic.**')


