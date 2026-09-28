import streamlit as st
import pandas as pd
import joblib


# Load the trained model
model = joblib.load("titanic_best_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")


# Passenger inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)


# Prediction button
if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Passenger is predicted to SURVIVE.")
    else:
        st.error("Passenger is predicted NOT to survive.")

    st.write(
        f"Survival probability: {probability:.2%}"
    )