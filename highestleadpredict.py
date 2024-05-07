import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'C:\\Users\\asunder\\LangChain\\Embeddings\\insurancehighestvaluelead\\saved_model'  # Replace with your model's path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Function to predict the interest for each sentence in the paragraph
def predict_lead_interest(text):
    sentences = text.split('.')  # Split the paragraph into sentences
    predictions = []
    for sentence in sentences:
        if sentence.strip():  # Check if the sentence is not empty
            inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits).item()
            predictions.append(predicted_class)
    return predictions

# Streamlit interface
st.title('Life Insurance Lead Prediction')
st.write('Enter the paragraph to predict if each sentence is a potential lead:')

# Text input
user_input = st.text_area("Text Input", "Type Here...")

if st.button('Predict'):
    # Make prediction
    predictions = predict_lead_interest(user_input)
    
    # Display predictions
    st.write('Predictions:')
    for i, prediction in enumerate(predictions):
        st.write(f'Sentence {i+1}: {"Potential Lead" if prediction == 1 else "Not a Potential Lead"}')
    
    # If you have the true labels, you can calculate and display the confusion matrix
    # true_labels = [...]  # Replace this with the actual list of true labels for each sentence
    # conf_matrix = confusion_matrix(true_labels, predictions)
    # st.table(conf_matrix)
