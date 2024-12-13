import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit App
st.title("Stock Price Prediction with LSTM")

# Function to build the model architecture
def build_lstm_model():
    """Recreate the LSTM model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(50, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load the model weights
@st.cache_resource
def load_model():
    try:
        model = build_lstm_model()
        model.load_weights("LSTM_SM_weights.h5")  # Make sure weights file is available
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
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
        return None, prediction[0][0], input_array
    except Exception as e:
        return f"Error during prediction: {e}", None, None

if model and input_data:
    error, prediction, input_array = make_prediction(input_data, model)
    if error:
        st.error(error)
    else:
        st.success(f"Predicted Stock Price (scaled): {prediction:.4f}")

        # Plot the input data and prediction
        fig, ax = plt.subplots()
        ax.plot(input_array[0], label="Input Stock Prices")
        ax.scatter([50], prediction, color="red", label="Predicted Price")
        ax.set_title("Stock Price Prediction")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Scaled Stock Prices")
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
