import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('logistic_regression_model.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input features based on X.columns from the notebook
Delivery_Distance = st.number_input('Delivery Distance', min_value=0.0, value=20.0)
Traffic_Congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
Weather_Condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
Delivery_Slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
Driver_Experience = st.number_input('Driver Experience (Years)', min_value=0, value=10)
Num_Stops = st.number_input('Number of Stops', min_value=0, value=5)
Vehicle_Age = st.number_input('Vehicle Age (Years)', min_value=0, value=5)
Road_Condition_Score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
Package_Weight = st.number_input('Package Weight', min_value=0.0, value=10.0)
Fuel_Efficiency = st.number_input('Fuel Efficiency', min_value=0.0, value=15.0)
Warehouse_Processing_Time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)

# Make prediction
if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input data (important for feature names)
    input_data = pd.DataFrame([[Delivery_Distance, Traffic_Congestion, Weather_Condition,
                                  Delivery_Slot, Driver_Experience, Num_Stops, Vehicle_Age,
                                  Road_Condition_Score, Package_Weight, Fuel_Efficiency,
                                  Warehouse_Processing_Time]],
                                columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                         'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                                         'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                                         'Warehouse_Processing_Time'])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f'Prediction: Delivery is likely to be Delayed (Probability: {prediction_proba[1]:.2f})')
    else:
        st.success(f'Prediction: Delivery is likely to be On Time (Probability: {prediction_proba[0]:.2f})')
