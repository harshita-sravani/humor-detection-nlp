# Humor Detection System (NLP-Based)

## Overview
This project implements an AI-based humor detection system using Natural Language Processing and Machine Learning techniques. The system classifies text as humorous or non-humorous and provides confidence scores through an interactive web interface.

## Why This Project
Humor detection is a challenging NLP problem due to its contextual and subjective nature. This project demonstrates how machine learning models can capture linguistic patterns to classify humor effectively, with applications in chatbots, content moderation, and social media analysis.

## Features
- Real-time humor detection from user input  
- TF-IDF based feature extraction  
- Machine learning models (SVM, Logistic Regression)  
- Interactive web interface using Gradio  
- Confidence-based predictions  
- Text preprocessing and normalization pipeline  

## Project Structure
- humor_detection_simple.py — Model training and NLP pipeline  
- gradio_app_simple.py — Web interface for real-time predictions  
- best_humor_model.pkl — Trained model  
- tfidf_vectorizer.pkl — Feature vectorizer  
- humor_detection_results.png — Model performance visualization  
- requirements.txt — Dependencies  

## Approach
- Preprocessed text using tokenization, stopword removal, and normalization  
- Extracted features using TF-IDF vectorization  
- Trained multiple models including Logistic Regression and SVM  
- Selected best model based on F1-score  
- Built a real-time Gradio interface for inference  

## Results
- Achieved up to 93% accuracy using SVM  
- Compared multiple models and selected best-performing model  
- Demonstrated strong performance on real-world text inputs  

## Results Visualization
![Model Results](humor_detection_results.png)

## Example
Input: "Why don’t scientists trust atoms? Because they make up everything!"  
Output: Humor detected with high confidence  

## Quick Start
pip install -r requirements.txt  
python gradio_app_simple.py  

Then open the local URL shown in the terminal.

## Tech Stack
- Python  
- NumPy, Pandas  
- scikit-learn  
- NLTK  
- Gradio  

## Dataset
Dataset is not included due to size constraints. Any labeled humor dataset can be used.

## Future Improvements
- Improve performance using transformer-based models  
- Deploy as a scalable web application  
- Extend to multi-class humor classification  
