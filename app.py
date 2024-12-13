import tensorflow as tf 
import numpy as np
import streamlit as st

# Streamlit App Code
st.title("Stock Price Prediction with LSTM")

# Load the trained model
try:
    model = tf.keras.models.load_model('LSTM_SM.keras')
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Sidebar Input
st.sidebar.header("Input Parameters")
input_data = st.sidebar.text_area(
    "Enter 50 scaled stock prices (comma-separated):", 
    placeholder="e.g., 0.0858, 0.0970, ... (50 values)"
)

if input_data:
    try:
        # Preprocessing the input data
        input_data = np.array([float(i) for i in input_data.split(",")])

        if len(input_data) != 50:
            st.error("Please provide exactly 50 values.")
        else:
            input_data = np.reshape(input_data, (1, 50, 1))
            
            # Make prediction
            try:
                prediction = model.predict(input_data)
                st.success(f"Predicted Stock Price (scaled): {prediction[0][0]:.4f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    except ValueError:
        st.error("Invalid input. Ensure all values are numeric and comma-separated.")



