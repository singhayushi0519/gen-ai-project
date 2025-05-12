"""
Next Sentence Prediction using Generative AI

This module contains the core functionality for predicting the next sentence
based on a given input context using pre-trained language models.
"""

import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    BertForNextSentencePrediction,
    BertTokenizer,
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline
)
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

class NextSentencePredictor:
    """Class to predict the next sentence based on input context."""
    
    def __init__(self, model_name="gpt2", use_gpu=False):
        """
        Initialize the predictor with a specific model.
        
        Args:
            model_name (str): Model identifier from HuggingFace 
                              (e.g., "gpt2", "gpt2-medium", "EleutherAI/gpt-neo-1.3B")
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.load_model(model_name)
        
    def load_model(self, model_name):
        """Load the specified model and tokenizer."""
        print(f"Loading model: {model_name}")
        
        if "gpt" in model_name.lower():
            # For GPT-style models (generative)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model_type = "generative"
            
        elif "bert" in model_name.lower():
            # For BERT-style models (next sentence prediction)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForNextSentencePrediction.from_pretrained(model_name)
            self.model.to(self.device)
            self.model_type = "nsp"
            
        else:
            # Default to a generative approach with automodel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model_type = "generative"
            
        print(f"Model loaded on {self.device}")
    
    def predict_next_sentence(self, context, num_predictions=3, max_length=50, temperature=0.7):
        """
        Predict the next sentence(s) based on the provided context.
        
        Args:
            context (str): The input text context
            num_predictions (int): Number of different predictions to generate
            max_length (int): Maximum length of each prediction in tokens
            temperature (float): Controls randomness in generation (lower = more deterministic)
            
        Returns:
            list: List of predicted next sentences
        """
        if self.model_type == "generative":
            return self._generate_continuations(context, num_predictions, max_length, temperature)
        else:
            raise NotImplementedError("Only generative models are currently supported")
    
    def _generate_continuations(self, context, num_predictions, max_length, temperature):
        """Generate text continuations and extract the next sentence."""
        # Encode the context
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Generate continuations
        generated_outputs = []
        
        for _ in range(num_predictions):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the new content (after the provided context)
            new_content = generated_text[len(context):].strip()
            
            # Try to extract just the first sentence from the new content
            new_sentences = sent_tokenize(new_content)
            if new_sentences:
                generated_outputs.append(new_sentences[0])
            else:
                generated_outputs.append(new_content)
        
        return generated_outputs
    
    def rank_predictions(self, context, predictions, method="perplexity"):
        """
        Rank the predicted sentences based on chosen method.
        
        Args:
            context (str): The original context
            predictions (list): List of predicted sentences
            method (str): Ranking method ('perplexity', 'coherence')
            
        Returns:
            list: Ranked predictions
        """
        if method == "perplexity":
            # Calculate perplexity for each prediction
            scores = []
            for pred in predictions:
                full_text = context + " " + pred
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    # Lower perplexity is better
                    perplexity = torch.exp(torch.tensor(loss)).item()
                    scores.append(perplexity)
            
            # Sort predictions by perplexity (lower is better)
            ranked_predictions = [pred for _, pred in sorted(zip(scores, predictions), key=lambda x: x[0])]
            return ranked_predictions
        
        else:
            # Default to returning the predictions without ranking
            return predictions


def evaluate_predictions(original_text, predicted_text):
    """
    Simple evaluation function to assess the quality of predictions.
    This could be expanded with more sophisticated metrics.
    
    Args:
        original_text (str): The ground truth next sentence
        predicted_text (str): The predicted next sentence
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Tokenize both texts
    original_tokens = original_text.lower().split()
    predicted_tokens = predicted_text.lower().split()
    
    # Calculate overlap (a simple metric)
    common_words = set(original_tokens).intersection(set(predicted_tokens))
    precision = len(common_words) / len(predicted_tokens) if predicted_tokens else 0
    recall = len(common_words) / len(original_tokens) if original_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "common_words": len(common_words)
    }


# Example usage
if __name__ == "__main__":
    # Create a predictor using the GPT-2 model
    predictor = NextSentencePredictor(model_name="gpt2")
    
    # Example context
    context = "The weather was perfect for a day at the beach. The sun was shining brightly."
    
    # Generate predictions
    predictions = predictor.predict_next_sentence(context, num_predictions=3)
    
    # Rank the predictions
    ranked_predictions = predictor.rank_predictions(context, predictions)
    
    # Display results
    print(f"Context: {context}")
    print("\nPredictions:")
    for i, pred in enumerate(ranked_predictions):
        print(f"{i+1}. {pred}")
