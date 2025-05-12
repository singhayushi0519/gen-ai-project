"""
Next Sentence Prediction - Web Interface

A Streamlit application that provides a user interface for the 
Next Sentence Prediction model.
"""

import streamlit as st
import torch
import nltk
from next_sentence_prediction import NextSentencePredictor

# Ensure NLTK punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Next Sentence Prediction",
    page_icon="📝",
    layout="wide"
)

# Application title and description
st.title("Next Sentence Prediction using Generative AI")
st.markdown("""
This application uses state-of-the-art language models to predict the next sentence 
based on the context you provide. Type or paste your text in the input box below,
and the AI will suggest what might come next.
""")

# Sidebar for model configuration
st.sidebar.header("Model Configuration")

model_options = {
    "GPT-2 (Small)": "gpt2",
    "GPT-2 (Medium)": "gpt2-medium",
    "DistilGPT-2": "distilgpt2"
}

# Select model
selected_model = st.sidebar.selectbox(
    "Choose a language model:",
    list(model_options.keys())
)

# Number of predictions
num_predictions = st.sidebar.slider(
    "Number of predictions to generate:",
    min_value=1,
    max_value=5,
    value=3
)

# Temperature for generation
temperature = st.sidebar.slider(
    "Temperature (creativity):",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1,
    help="Higher values produce more diverse results, lower values are more focused and deterministic"
)

# Max length for generated text
max_length = st.sidebar.slider(
    "Maximum length of prediction (tokens):",
    min_value=10,
    max_value=100,
    value=30
)

# Use GPU if available
use_gpu = st.sidebar.checkbox(
    "Use GPU (if available)",
    value=torch.cuda.is_available()
)

# Create the text input area
user_input = st.text_area(
    "Enter your text context here:",
    height=150,
    placeholder="The weather was perfect for a day at the beach. The sun was shining brightly."
)

# Initialize session state for the predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.current_model = None

# Create or update the predictor when needed
def get_predictor():
    model_name = model_options[selected_model]
    
    # Only create a new predictor if the model changes
    if (st.session_state.predictor is None or 
            model_name != st.session_state.current_model):
        
        with st.spinner(f"Loading {selected_model} model... This may take a moment."):
            st.session_state.predictor = NextSentencePredictor(
                model_name=model_name,
                use_gpu=use_gpu
            )
            st.session_state.current_model = model_name
    
    return st.session_state.predictor

# Process the input when the user clicks the button
if st.button("Predict Next Sentence"):
    if user_input:
        try:
            # Get or create the predictor
            predictor = get_predictor()
            
            # Generate predictions
            with st.spinner("Generating predictions..."):
                predictions = predictor.predict_next_sentence(
                    user_input,
                    num_predictions=num_predictions,
                    max_length=max_length,
                    temperature=temperature
                )
                
                # Rank the predictions if more than one
                if len(predictions) > 1:
                    ranked_predictions = predictor.rank_predictions(user_input, predictions)
                else:
                    ranked_predictions = predictions
            
            # Display results
            st.subheader("Predicted Next Sentences:")
            for i, pred in enumerate(ranked_predictions):
                st.markdown(f"**{i+1}.** {pred}")
                
                # Add a button to copy the prediction
                if st.button(f"Copy prediction #{i+1}", key=f"copy_{i}"):
                    st.session_state.copied_text = pred
                    st.success(f"Prediction #{i+1} copied to clipboard!")
            
            # Display the full text with the best prediction
            if ranked_predictions:
                st.subheader("Complete Text with Best Prediction:")
                complete_text = f"{user_input} {ranked_predictions[0]}"
                st.markdown(f"> {complete_text}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text to generate predictions.")

# Footer information
st.markdown("---")
st.markdown("""
### About This Project
This application uses transformer-based language models to predict the next sentence in a sequence. 
It can be useful for:
- Content generation assistance
- Story writing and brainstorming
- Testing language model capabilities

Built with Streamlit, Transformers, and PyTorch.
""")
