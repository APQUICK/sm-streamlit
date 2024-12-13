import tensorflow as tf
import numpy as np
import streamlit as st



# Streamlit App Code
def main():
    st.title("Stock Price Prediction")

    # Load the trained model
    model = tf.keras.models.load_model('LSTM_SM.keras')

    st.sidebar.header("Input Parameters")

    # User input for stock data (scaled values)
    input_data = st.sidebar.text_area(
        "Enter 50 scaled stock prices (comma-separated):", "0.0858, 0.0970, ..."
    )

    if input_data:
        try:
            # Preprocessing the input data
            input_data = np.array([float(i) for i in input_data.split(",")])

            if len(input_data) != 50:
                st.error("Please provide exactly 50 values.")
                return

            input_data = np.reshape(input_data, (1, 50, 1))

            # Make prediction
            prediction = model.predict(input_data)

            st.success(f"Predicted Stock Price (scaled): {prediction[0][0]:.4f}")

        except ValueError:
            st.error("Invalid input. Ensure all values are numeric and comma-separated.")

if _name_ == "_main_":
    main()
