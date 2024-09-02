import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
tokenizer_path = r'C:\Users\HP\Desktop\Data Science\EXCLER Projects\NLP Sentiment and Classification Analysis\tokenizer.pkl'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the LSTM and GRU models with exception handling
lstm_model_path = r'C:\Users\HP\Desktop\Data Science\EXCLER Projects\NLP Sentiment and Classification Analysis\lstm_model.h5'
gru_model_path = r'C:\Users\HP\Desktop\Data Science\EXCLER Projects\NLP Sentiment and Classification Analysis\gru_model.h5'

try:
    lstm_model = load_model(lstm_model_path)
except Exception as e:
    st.error(f"Error loading LSTM model: {str(e)}")

try:
    gru_model = load_model(gru_model_path)
except Exception as e:
    st.error(f"Error loading GRU model: {str(e)}")

# Function to predict sentiment
def predict_sentiment(text, model):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    max_length = 100  # Ensure this matches the training max_length
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    # Interpret the result
    if prediction >= 0.5:
        return "Positive", prediction
    else:
        return "Negative", prediction

# Streamlit app
st.title("Customer Reviews Sentiment Analysis with LSTM and GRU")

# Input text box
user_input = st.text_area("Enter text for sentiment analysis:")

# Dropdown to select the model
model_option = st.selectbox("Choose the model", ("LSTM", "GRU"))

# Predict button
if st.button("Predict Sentiment"):
    if user_input:
        if model_option == "LSTM":
            sentiment, confidence = predict_sentiment(user_input, lstm_model)
        else:
            sentiment, confidence = predict_sentiment(user_input, gru_model)
        
        # Display results
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter text for analysis.")
