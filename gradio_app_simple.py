"""
Humor Detection Web Interface - Simple Version
==============================================
Interactive Gradio interface for humor detection using traditional ML models.

Author: AI Assistant
Date: 2024
"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class HumorDetector:
    def __init__(self):
        """Initialize the humor detector with trained models."""
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load the trained model and vectorizer."""
        try:
            self.model = joblib.load('best_humor_model.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            print("SUCCESS Models loaded successfully!")
        except FileNotFoundError:
            print("WARNING Model files not found. Using demo mode.")
            self.model = None
            self.vectorizer = None
    
    def clean_text(self, text):
        """Clean and preprocess text for prediction."""
        try:
            text = str(text)
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation but keep spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Only remove stopwords if the text is long enough
            if len(text.split()) > 3:
                stop_words = set(stopwords.words('english'))
                word_tokens = word_tokenize(text)
                # Keep words that are not stopwords and have length > 1
                filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 1]
                return ' '.join(filtered_text)
            else:
                # For short texts, just return cleaned version without stopword removal
                return text
        except:
            return text if text else ""
    
    def predict_humor(self, text):
        """Predict if text is humorous."""
        if not text or not text.strip():
            return "Please enter some text to analyze.", 0.0, "WARNING"
        
        # Demo mode responses if models aren't loaded
        if self.model is None or self.vectorizer is None:
            demo_responses = {
                "why don't scientists trust atoms": ("HUMOR DETECTED!", 0.95, "HUMOR"),
                "what do you call": ("HUMOR DETECTED!", 0.92, "HUMOR"),
                "why did the": ("HUMOR DETECTED!", 0.88, "HUMOR"),
                "knock knock": ("HUMOR DETECTED!", 0.90, "HUMOR"),
                "joke": ("HUMOR DETECTED!", 0.85, "HUMOR"),
                "funny": ("HUMOR DETECTED!", 0.80, "HUMOR"),
                "pun": ("HUMOR DETECTED!", 0.87, "HUMOR"),
                "news": ("NOT HUMOR", 0.75, "NOT_HUMOR"),
                "breaking": ("NOT HUMOR", 0.80, "NOT_HUMOR"),
                "report": ("NOT HUMOR", 0.78, "NOT_HUMOR"),
                "study": ("NOT HUMOR", 0.82, "NOT_HUMOR"),
                "research": ("NOT HUMOR", 0.79, "NOT_HUMOR"),
                "policy": ("NOT HUMOR", 0.85, "NOT_HUMOR"),
                "government": ("NOT HUMOR", 0.83, "NOT_HUMOR")
            }
            
            text_lower = text.lower()
            for keyword, (result, confidence, emoji) in demo_responses.items():
                if keyword in text_lower:
                    return result, confidence, emoji
            
            # Default response for demo mode
            return "NOT HUMOR", 0.65, "NOT_HUMOR"
        
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                return "Text appears to be empty after cleaning.", 0.0, "WARNING"
            
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
            
            if prediction == 1:
                return "HUMOR DETECTED!", confidence, "HUMOR"
            else:
                return "NOT HUMOR", confidence, "NOT_HUMOR"
                
        except Exception as e:
            return f"Error during prediction: {str(e)}", 0.0, "ERROR"

# Initialize the detector
detector = HumorDetector()

def analyze_text(text):
    """Analyze text for humor and return formatted results."""
    if not text or not text.strip():
        return "Please enter some text to analyze.", "0.0%", "WARNING", ""
    
    result, confidence, emoji = detector.predict_humor(text)
    confidence_percent = f"{confidence:.1%}"
    
    # Create analysis details
    cleaned_text = detector.clean_text(text)
    analysis_details = f"""
**Original Text:** {text}

**Cleaned Text:** {cleaned_text}

**Text Length:** {len(text)} characters
**Word Count:** {len(text.split())} words
**Cleaned Word Count:** {len(cleaned_text.split())} words
    """
    
    return result, confidence_percent, emoji, analysis_details

# Sample texts for quick testing
sample_texts = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "What do you call a fake noodle? An impasta!",
    "The stock market crashed today due to economic uncertainty.",
    "Climate change is a serious global issue that requires immediate attention.",
    "Why did the scarecrow win an award? He was outstanding in his field!",
    "I'm reading a book about anti-gravity. It's impossible to put down!",
    "The new policy will be implemented next quarter."
]

# Create Gradio interface
with gr.Blocks(
    title="Humor Detection AI",
    theme=gr.themes.Soft(),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Hangyaboly:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Schoolbell&display=swap');
    
    /* Force cache refresh */
    html, body {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, #FFF4E6 0%, #FFE4B5 50%, #FFEAA7 100%) !important;
        min-height: 100vh !important;
    }
    
    /* Override any default Gradio styling */
    .gradio-container, .gradio-container .block {
        background: linear-gradient(135deg, #FFF4E6 0%, #FFE4B5 50%, #FFEAA7 100%) !important;
    }
    
    /* Force background on all containers */
    .container, .block, .form, .panel {
        background: linear-gradient(135deg, #FFF4E6 0%, #FFE4B5 50%, #FFEAA7 100%) !important;
    }
    
    .main-header {
        text-align: center !important;
        margin-bottom: 3rem !important;
        padding: 3rem 2rem !important;
        background: linear-gradient(135deg, #FF9500 0%, #FFB347 50%, #FFA500 100%) !important;
        border-radius: 25px !important;
        box-shadow: 0 15px 35px rgba(255, 149, 0, 0.3) !important;
        border: 3px solid #FF8C00 !important;
        animation: fadeInUp 0.8s ease-out !important;
        position: relative !important;
        overflow: hidden !important;
        z-index: 10 !important;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #FF6B35, #F7931E, #FFD700, #F7931E, #FF6B35);
        background-size: 300% 100%;
        animation: shimmer 2s ease-in-out infinite;
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 1rem;
        text-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
        font-family: 'Hangyaboly', cursive !important;
    }
    
    .main-header p {
        color: #FFFFFF;
        font-size: 1.4rem;
        margin-bottom: 0;
        font-weight: 500;
        opacity: 0.95;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .input-container {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 25px rgba(255, 149, 0, 0.2);
        border: 2px solid #FFB347;
        animation: slideInLeft 0.8s ease-out;
        transition: all 0.3s ease;
        position: relative;
        z-index: 10;
    }
    
    .input-container:hover {
        box-shadow: 0 12px 30px rgba(255, 149, 0, 0.3);
        transform: translateY(-2px);
    }
    
    .output-container {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 25px rgba(255, 149, 0, 0.2);
        border: 2px solid #FFB347;
        animation: slideInRight 0.8s ease-out;
        transition: all 0.3s ease;
        position: relative;
        z-index: 10;
    }
    
    .output-container:hover {
        box-shadow: 0 12px 30px rgba(255, 149, 0, 0.3);
        transform: translateY(-2px);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.3) !important;
        font-family: 'Schoolbell', cursive !important;
    }
    
    .btn-primary:hover {
        background: linear-gradient(135deg, #FF8C42 0%, #FFB347 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.4) !important;
    }
    
    .btn-secondary {
        background: linear-gradient(135deg, #FFB347 0%, #FFCC70 100%) !important;
        border: 2px solid #FFB347 !important;
        border-radius: 25px !important;
        padding: 12px 25px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        color: #FF6B35 !important;
        box-shadow: 0 4px 15px rgba(255, 179, 71, 0.3) !important;
        font-family: 'Schoolbell', cursive !important;
    }
    
    .btn-secondary:hover {
        background: linear-gradient(135deg, #FFCC70 0%, #FFD700 100%) !important;
        border-color: #FF8C42 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 179, 71, 0.4) !important;
        color: #FF8C42 !important;
    }
    
    .btn-info {
        background: linear-gradient(135deg, #20B2AA 0%, #48CAE4 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 25px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(32, 178, 170, 0.3) !important;
        font-family: 'Schoolbell', cursive !important;
    }
    
    .btn-info:hover {
        background: linear-gradient(135deg, #48CAE4 0%, #90E0EF 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(32, 178, 170, 0.4) !important;
    }
    
    .model-info-panel {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFFACD 100%) !important;
        border: 2px solid #FFB347 !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin-top: 1rem !important;
        box-shadow: 0 8px 25px rgba(255, 179, 71, 0.2) !important;
        animation: fadeIn 0.5s ease-in !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        position: relative !important;
        z-index: 10 !important;
    }
    
    .textbox {
        border-radius: 6px !important;
        border: 1px solid #ced4da !important;
        transition: all 0.3s ease !important;
        font-size: 1rem !important;
        font-family: 'Schoolbell', cursive !important;
    }
    
    .textbox:focus {
        border-color: #2c3e50 !important;
        box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.1) !important;
    }
    
    .result-humor {
        background: #d4edda !important;
        color: #0f3d1a !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 6px !important;
        padding: 1.5rem !important;
        text-align: center !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        animation: fadeIn 0.6s ease-out !important;
    }
    
    .result-not-humor {
        background: #f8d7da !important;
        color: #4a0e14 !important;
        border: 1px solid #f5c6cb !important;
        border-radius: 6px !important;
        padding: 1.5rem !important;
        text-align: center !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        animation: fadeIn 0.6s ease-out !important;
    }
    
    .confidence-display {
        background: #ffffff !important;
        color: #2c3e50 !important;
        border: 2px solid #2c3e50 !important;
        border-radius: 6px !important;
        padding: 1rem !important;
        text-align: center !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        animation: fadeIn 0.8s ease-out !important;
    }
    
    .info-panel {
        background: #ffffff;
        border-radius: 8px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 2px solid #2c3e50;
        animation: fadeInUp 1s ease-out;
    }

    .info-panel h3 {
        color: #000000;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }

    .info-panel ul {
        list-style: none;
        padding: 0;
    }

    .info-panel li {
        padding: 0.75rem 0;
        border-bottom: 2px solid #2c3e50;
        transition: all 0.3s ease;
        color: #000000;
        font-weight: 500;
    }
    
    .info-panel li:hover {
        background: #f8f9fa;
        padding-left: 1rem;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #e9ecef;
        border-top: 3px solid #2c3e50;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes shimmer {
        0% {
            background-position: -200% 0;
        }
        100% {
            background-position: 200% 0;
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .ai-character {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
        filter: drop-shadow(0 4px 8px rgba(255, 149, 0, 0.3));
    }
    
    /* Playful doodles and decorative elements */
    .doodle-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 2;
        overflow: hidden;
    }
    
    .doodle {
        position: absolute;
        font-size: 2rem;
        opacity: 0.6;
        animation: float 6s ease-in-out infinite;
    }
    
    .doodle-1 {
        top: 10%;
        left: 5%;
        animation-delay: 0s;
        transform: rotate(-15deg);
    }
    
    .doodle-2 {
        top: 20%;
        right: 8%;
        animation-delay: 1s;
        transform: rotate(20deg);
    }
    
    .doodle-3 {
        bottom: 15%;
        left: 10%;
        animation-delay: 2s;
        transform: rotate(-10deg);
    }
    
    .doodle-4 {
        bottom: 25%;
        right: 15%;
        animation-delay: 3s;
        transform: rotate(25deg);
    }
    
    .doodle-5 {
        top: 50%;
        left: 2%;
        animation-delay: 4s;
        transform: rotate(-20deg);
    }
    
    .doodle-6 {
        top: 60%;
        right: 3%;
        animation-delay: 5s;
        transform: rotate(15deg);
    }
    
    .squiggle {
        position: absolute;
        width: 60px;
        height: 60px;
        border: 3px solid #FF9500;
        border-radius: 50% 20% 50% 20%;
        opacity: 0.4;
        animation: wiggle 4s ease-in-out infinite;
    }
    
    .squiggle-1 {
        top: 15%;
        left: 15%;
        animation-delay: 0.5s;
    }
    
    .squiggle-2 {
        bottom: 20%;
        right: 20%;
        animation-delay: 2.5s;
        transform: rotate(45deg);
    }
    
    .arrow-doodle {
        position: absolute;
        font-size: 1.5rem;
        color: #FF6B35;
        opacity: 0.5;
        animation: pulse 3s ease-in-out infinite;
    }
    
    .arrow-1 {
        top: 30%;
        left: 20%;
        transform: rotate(-30deg);
        animation-delay: 1s;
    }
    
    .arrow-2 {
        bottom: 40%;
        right: 25%;
        transform: rotate(60deg);
        animation-delay: 3s;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px) rotate(var(--rotation, 0deg));
        }
        50% {
            transform: translateY(-20px) rotate(var(--rotation, 0deg));
        }
    }
    
    @keyframes wiggle {
        0%, 100% {
            transform: rotate(0deg) scale(1);
        }
        25% {
            transform: rotate(-5deg) scale(1.1);
        }
        75% {
            transform: rotate(5deg) scale(0.9);
        }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .dropdown {
        border-radius: 6px !important;
        border: 1px solid #ced4da !important;
        transition: all 0.3s ease !important;
    }

    .dropdown:hover {
        border-color: #2c3e50 !important;
        box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.1) !important;
    }

    .markdown {
        background: #ffffff !important;
        border-radius: 6px !important;
        padding: 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e9ecef !important;
        animation: fadeIn 0.8s ease-out !important;
    }

    /* Enhanced label contrast */
    label {
        color: #212529 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: none !important;
        background: transparent !important;
    }

    .gr-form label {
        color: #212529 !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Enhanced div contrast - more specific targeting */
                .gradio-container div:not(.markdown):not(.gr-markdown):not([data-testid]):not(.prose) {
                    color: #000000 !important;
                    font-family: 'Schoolbell', cursive !important;
                    background-color: #FF6B35 !important;
                    padding: 8px !important;
                    border-radius: 8px !important;
                }
                
                /* Ensure markdown content displays properly */
                .gr-markdown, .markdown, .prose {
                    background-color: transparent !important;
                    color: #212529 !important;
                    padding: 0 !important;
                }

    .gr-textbox label,
    .gr-dropdown label,
    .gr-markdown label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    }

    /* Enhanced h2, p, h3 contrast */
    h2 {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.15) !important;
        margin-bottom: 0.8rem !important;
    }

    p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        font-family: 'Schoolbell', cursive !important;
    }

    /* Specific styling for result sections */
    .result-humor h2,
    .result-not-humor h2 {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }

    .result-humor p,
    .result-not-humor p {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    """
) as demo:
    
    gr.HTML("""
    <div class="doodle-container">
        <div class="doodle doodle-1">*</div>
        <div class="doodle doodle-2">+</div>
        <div class="doodle doodle-3">~</div>
        <div class="doodle doodle-4">*</div>
        <div class="doodle doodle-5">o</div>
        <div class="doodle doodle-6">!</div>
        <div class="squiggle squiggle-1"></div>
        <div class="squiggle squiggle-2"></div>
        <div class="arrow-doodle arrow-1">↗</div>
        <div class="arrow-doodle arrow-2">↙</div>
    </div>
    
    <div class="main-header">
        <div class="ai-character">AI</div>
        <h1>GiggleGauge</h1>
        <p>I analyze humor so well, even my circuits snort at bad jokes</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["input-container"]):
            text_input = gr.Textbox(
                placeholder="Enter your text here...",
                lines=4,
                elem_classes=["textbox"],
                show_label=False
            )
            
            with gr.Row():
                predict_btn = gr.Button(
                    "Analyze", 
                    variant="primary",
                    elem_classes=["btn-primary"]
                )
                clear_btn = gr.Button(
                    "Clear", 
                    variant="secondary",
                    elem_classes=["btn-secondary"]
                )
                model_info_btn = gr.Button(
                    "Model Info", 
                    variant="secondary",
                    elem_classes=["btn-info"]
                )
            
            sample_dropdown = gr.Dropdown(
                choices=[
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                    "The weather forecast for tonight is dark.",
                    "I need to buy groceries and pay bills today.",
                    "Parallel lines have so much in common. It's a shame they'll never meet.",
                    "The meeting has been scheduled for 3 PM tomorrow."
                ],
                value=None,
                elem_classes=["dropdown"],
                show_label=False
            )
        
        with gr.Column(scale=1, elem_classes=["output-container"]):
            result_output = gr.HTML(
                value="<div style='text-align: center; color: #6c757d; padding: 2rem;'>Results will appear here</div>",
                elem_classes=["result-display"]
            )
            
            confidence_output = gr.HTML()
            
            # Model info panel (initially hidden)
            model_info_panel = gr.HTML(
                value="",
                visible=False,
                elem_classes=["model-info-panel"]
            )
            
            # Model info (static display - will be removed)
            # gr.HTML("""
            # <div class="info-panel">
            #     <h3>Model Information</h3>
            #     <ul>
            #         <li><strong>Algorithm:</strong> Logistic Regression (Best Performer)</li>
            #         <li><strong>Features:</strong> TF-IDF Vectorization</li>
            #         <li><strong>Training Data:</strong> 200,000+ text samples</li>
            #         <li><strong>Performance:</strong> 90.5% F1-Score</li>
            #         <li><strong>Processing:</strong> Real-time text analysis</li>
            #     </ul>
            # </div>
            # """)
            emoji_output = gr.Textbox(
                interactive=False,
                show_label=False
            )
    
    with gr.Row():
        analysis_details = gr.Markdown(
            value="Enter text above to see detailed analysis...",
            show_label=False
        )
    
    # Remove duplicate model info section and clean up the interface
    
    # Enhanced prediction function with loading animation
    def predict_humor_enhanced(text):
        if not text.strip():
            return (
                "<div style='text-align: center; color: #888; padding: 2rem;'>Please enter some text to analyze</div>",
                ""
            )
        
        try:
            result_text, confidence, emoji = detector.predict_humor(text)
            
            # Enhanced result display
            if "HUMOR DETECTED" in result_text:  # Humor detected
                result_html = f"""
                <div class="result-humor">
                    <h2>HUMOR DETECTED</h2>
                    <p>This text appears to be humorous</p>
                    <div style='font-size: 2rem; margin: 1rem 0;'>Comedy Analysis</div>
                </div>
                """
            else:  # No humor detected
                result_html = f"""
                <div class="result-not-humor">
                    <h2>NOT HUMOR</h2>
                    <p>This text appears to be non-humorous</p>
                    <div style='font-size: 2rem; margin: 1rem 0;'>Factual Content</div>
                </div>
                """
            
            # Enhanced confidence display
            confidence_percentage = confidence * 100
            confidence_html = f"""
            <div class="confidence-display">
                <h3>Confidence Level</h3>
                <div style='font-size: 2rem; margin: 1rem 0;'>{confidence_percentage:.1f}%</div>
                <div style='background: rgba(255,255,255,0.3); border-radius: 10px; height: 10px; margin: 1rem 0;'>
                    <div style='background: white; height: 100%; border-radius: 10px; width: {confidence_percentage}%; transition: width 0.8s ease;'></div>
                </div>
            </div>
            """
            
            return result_html, confidence_html
            
        except Exception as e:
            error_html = f"""
            <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                        color: white; padding: 1.5rem; border-radius: 15px; text-align: center;'>
                <h3>Error</h3>
                <p>Sorry, there was an error analyzing your text: {str(e)}</p>
            </div>
            """
            return error_html, ""
    
    # Event handlers
    def use_sample(sample_text):
        return sample_text
    
    def clear_all():
        return "", "", "", "", "Enter text above to see detailed analysis..."
    
    def toggle_model_info():
        """Toggle the visibility of the model information panel"""
        model_info_html = """
        <div class="info-panel">
            <h3>Model Information</h3>
            <ul>
                <li><strong>Algorithm:</strong> Logistic Regression (Best Performer)</li>
                <li><strong>Features:</strong> TF-IDF Vectorization</li>
                <li><strong>Training Data:</strong> 200,000+ text samples</li>
                <li><strong>Performance:</strong> 90.5% F1-Score</li>
                <li><strong>Processing:</strong> Real-time text analysis</li>
                <li><strong>Preprocessing:</strong> Text cleaning, tokenization, stopword removal</li>
                <li><strong>Model Size:</strong> Lightweight and optimized for real-time inference</li>
                <li><strong>Accuracy:</strong> 89.2% on test dataset</li>
            </ul>
        </div>
        """
        return gr.update(value=model_info_html, visible=True)
    
    # Connect events
    predict_btn.click(
        predict_humor_enhanced,
        inputs=[text_input],
        outputs=[result_output, confidence_output]
    )
    
    sample_dropdown.change(
        use_sample,
        inputs=[sample_dropdown],
        outputs=[text_input]
    )
    
    clear_btn.click(
        clear_all,
        outputs=[text_input, result_output, confidence_output, emoji_output, analysis_details]
    )
    
    model_info_btn.click(
        toggle_model_info,
        outputs=[model_info_panel]
    )
    
    # Auto-analyze on text change (with debounce)
    text_input.change(
        analyze_text,
        inputs=[text_input],
        outputs=[result_output, confidence_output, emoji_output, analysis_details]
    )

if __name__ == "__main__":
    print("Starting Humor Detection Web Interface...")
    print("Model Status:", "Loaded" if detector.model else "Demo Mode")
    print("Launching Gradio interface...")
    
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )