import tensorflow as tf
import numpy as np
import streamlit as st

# Streamlit App
st.title("Stock Price Prediction with LSTM")

# Load the trained LSTM model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("LSTM_SM.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for user input
st.sidebar.header("Input Parameters")
input_data = st.sidebar.text_area(
    "Enter the last 50 scaled stock prices (comma-separated):",
    placeholder="e.g., 0.0858, 0.0970, ... (50 values)"
)

# Function to preprocess and predict
def make_prediction(input_string, model):
    try:
        # Convert input string to numpy array
        input_array = np.array([float(x.strip()) for x in input_string.split(",")])
        
        # Ensure exactly 50 values are provided
        if len(input_array) != 50:
            return "Invalid input: Please provide exactly 50 values.", None

        # Reshape input for the model
        input_array = np.reshape(input_array, (1, 50, 1))
        
        # Predict using the model
        prediction = model.predict(input_array)
        return None, prediction[0][0]  # Return prediction
    except Exception as e:
        return f"Error during prediction: {e}", None

if model and input_data:
    error, prediction = make_prediction(input_data, model)
    if error:
        st.error(error)
    else:
        st.success(f"Predicted Stock Price (scaled): {prediction:.4f}")




