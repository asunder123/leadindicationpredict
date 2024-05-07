import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('insurance_lead_model.joblib')

# Set up the Streamlit app
st.title("Insurance Lead Prediction App")
st.write("Enter your information and a conversation excerpt to predict if it's a potential lead.")

# User input fields
age = st.slider("Your Age", min_value=18, max_value=100, value=30)
income = st.number_input("Your Income", min_value=0, value=50000)
conversation = st.text_area("Conversation Excerpt", "Considering life insurance for my new business loan.")

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Conversation': [conversation]  # Ensure this matches the training data
})

# Debugging: Print input data to check
st.write("Input data for prediction:", input_data)

# Make predictions
prediction = model.predict(input_data)[0]

# Debugging: Print prediction result
st.write("Prediction result:", prediction)

# Display prediction result
if prediction == 1:
    st.success("Potential Lead! 🚀")
elif prediction == 0:
    st.info("Not a Potential Lead")

# Optional: Display model confidence scores or additional insights
confidence = model.predict_proba(input_data)[0][1]
st.write(f"Confidence: {confidence:.2f}")
