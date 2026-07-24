import streamlit as st
import pandas as pd
import joblib
import os

model = joblib.load("LR_insurance.pkl")
scaler = joblib.load("insurance_scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Insurance Cost Predictor",
    layout="wide"
)

st.sidebar.title("Insurance Cost Prediction")
st.sidebar.write("Optional Assignment 20")
st.sidebar.write("Linear Regression Model")
st.sidebar.write("Developed using Streamlit")

st.title("🏥 Insurance Cost Prediction System")
st.write("Enter the details below to predict insurance expenses.")

st.divider()

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

predict = st.button("Predict Insurance Cost")

if predict:

    input_df = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]
    })

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=encoded_columns, fill_value=0)

    numerical_cols = ["age", "bmi", "children"]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prediction = model.predict(input_df)

    st.success(f"Predicted Insurance Cost : ₹{prediction[0]:,.2f}")

st.divider()

st.caption("Developed using Streamlit | AIML Optional Assignment 20")

