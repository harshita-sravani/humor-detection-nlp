"""
Humor Detection NLP Model - Demo Version
========================================
Fast demo version using a subset of data to demonstrate all functionality.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

print("Starting Humor Detection Model Training (Demo Version)...")

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer already available")
except LookupError:
    nltk.download('punkt')
    print("Punkt tokenizer downloaded")

try:
    nltk.data.find('corpora/stopwords')
    print("Stopwords already available")
except LookupError:
    nltk.download('stopwords')
    print("Stopwords downloaded")

# Print header
print("=" * 60)
print("HUMOR DETECTION NLP MODEL - DEMO VERSION")
print("=" * 60)

# 1. Load and explore data
print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

try:
    df = pd.read_csv('dataset.csv')
    print(f"Dataset loaded successfully!")
    
    # Sample the data for demo (use smaller subset)
    sample_size = min(1000, len(df))  # Use 1000 samples or less if dataset is smaller
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset shape: {df_sample.shape}")
    print(f"Columns: {list(df_sample.columns)}")
    print(f"\nClass distribution:")
    print(df_sample['humor'].value_counts())
    print(f"\nSample texts:")
    for i, row in df_sample.head(3).iterrows():
        humor_status = "HUMOR" if row['humor'] else "NOT HUMOR"
        print(f"- [{humor_status}] {row['text'][:100]}...")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating sample data for demo...")
    
    # Create sample data if file not found
    sample_data = {
        'text': [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "What do you call a fake noodle? An impasta!",
            "The stock market experienced significant volatility today.",
            "Climate change remains a pressing global concern.",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "The new policy will be implemented next quarter.",
            "I'm reading a book about anti-gravity. It's impossible to put down!"
        ],
        'humor': [True, True, True, False, False, True, False, True]
    }
    df_sample = pd.DataFrame(sample_data)
    print(f"Sample dataset created with {len(df_sample)} examples")

# 2. Text preprocessing
def clean_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

print("\n2. TEXT PREPROCESSING")
print("-" * 40)
print("Cleaning text data...")

df_sample['cleaned_text'] = df_sample['text'].apply(clean_text)
print(f"Text cleaning completed")

# Show sample of cleaned text
print("\nSample of cleaned text:")
for i, (original, cleaned) in enumerate(zip(df_sample['text'].head(3), df_sample['cleaned_text'].head(3))):
    print(f"\nExample {i+1}:")
    print(f"Original: {original[:80]}...")
    print(f"Cleaned:  {cleaned[:80]}...")

print("\n3. TRAIN-TEST SPLIT (80/20)")
print("-" * 40)

# Split the data
X = df_sample['cleaned_text']
y = df_sample['humor']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Data split completed")

# 4. Feature extraction
print("\n4. FEATURE EXTRACTION - TF-IDF")
print("-" * 40)

print("Extracting TF-IDF features...")

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF extraction completed")
print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# 5. Model training
print("\n5. MODEL TRAINING")
print("-" * 40)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

trained_models = {}
results = {}

# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = models['Logistic Regression']
lr_model.fit(X_train_tfidf, y_train)
print("Logistic Regression trained")

# Train SVM
print("Training SVM...")
svm_model = models['SVM']
svm_model.fit(X_train_tfidf, y_train)
print("SVM trained")

print("\n6. EVALUATING MODELS")
print("-" * 40)

for model_name, model in [('Logistic Regression', lr_model), ('SVM', svm_model)]:
    print(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model': model
    }
    
    trained_models[model_name] = model

print("\n7. MODEL COMPARISON")
print("-" * 40)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1_score'] for model in results.keys()]
})

print(comparison_df.round(4))

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
print(f"\nBest performing model: {best_model_name}")
print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")

print("\n8. CREATING VISUALIZATIONS")
print("-" * 40)

try:
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Humor Detection Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. Model Performance Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x + i*width, comparison_df[metric], width, label=metric, alpha=0.8)
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(comparison_df['Model'], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix for best model
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred_best)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Class Distribution
    class_counts = df_sample['humor'].value_counts()
    axes[0, 2].pie(class_counts.values, labels=['Not Humor', 'Humor'], autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Class Distribution in Dataset')
    
    # 4. Word Cloud for Humorous Text
    humor_text = ' '.join(df_sample[df_sample['humor'] == True]['cleaned_text'])
    if humor_text.strip():
        wordcloud_humor = WordCloud(width=400, height=300, background_color='white').generate(humor_text)
        axes[1, 0].imshow(wordcloud_humor, interpolation='bilinear')
        axes[1, 0].axis('off')
        axes[1, 0].set_title('WordCloud - Humorous Text')
    
    # 5. Word Cloud for Non-Humorous Text
    non_humor_text = ' '.join(df_sample[df_sample['humor'] == False]['cleaned_text'])
    if non_humor_text.strip():
        wordcloud_non_humor = WordCloud(width=400, height=300, background_color='white').generate(non_humor_text)
        axes[1, 1].imshow(wordcloud_non_humor, interpolation='bilinear')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('WordCloud - Non-Humorous Text')
    
    # 6. Text Length Distribution
    df_sample['text_length'] = df_sample['text'].str.len()
    humor_lengths = df_sample[df_sample['humor'] == True]['text_length']
    non_humor_lengths = df_sample[df_sample['humor'] == False]['text_length']
    
    axes[1, 2].hist(humor_lengths, alpha=0.7, label='Humor', bins=20)
    axes[1, 2].hist(non_humor_lengths, alpha=0.7, label='Not Humor', bins=20)
    axes[1, 2].set_xlabel('Text Length (characters)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Text Length Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('humor_detection_results.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'humor_detection_results.png'")
    plt.close()

except Exception as e:
    print(f"Warning: Could not create all visualizations: {e}")

print("\n9. SAVING BEST MODEL")
print("-" * 40)

try:
    # Save the best model and vectorizer
    joblib.dump(results[best_model_name]['model'], 'best_humor_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print(f"{best_model_name} model saved as 'best_humor_model.pkl'")
    print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
except Exception as e:
    print(f"Error saving model: {e}")

print("\n10. SAMPLE PREDICTIONS")
print("-" * 40)

# Test with sample texts
test_texts = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "The stock market crashed today due to economic uncertainty.",
    "What do you call a fake noodle? An impasta!",
    "Climate change is a serious global issue.",
    "I told my wife she was drawing her eyebrows too high. She looked surprised."
]

print("Sample Predictions:")
print("-" * 20)

for i, text in enumerate(test_texts, 1):
    try:
        cleaned = clean_text(text)
        text_tfidf = tfidf.transform([cleaned])
        prediction = results[best_model_name]['model'].predict(text_tfidf)[0]
        probability = results[best_model_name]['model'].predict_proba(text_tfidf)[0]
        confidence = max(probability)
        
        humor_label = "HUMOR" if prediction == 1 else "NOT HUMOR"
        
        print(f"{i}. Text: {text[:60]}...")
        print(f"   Prediction: {humor_label}")
        print(f"   Confidence: {confidence:.3f}")
        print()
        
    except Exception as e:
        print(f"Error predicting for text {i}: {e}")

print("\n11. FEATURE ANALYSIS")
print("-" * 40)

try:
    # Get feature names and coefficients for Logistic Regression
    if 'Logistic Regression' in results:
        feature_names = tfidf.get_feature_names_out()
        coefficients = results['Logistic Regression']['model'].coef_[0]
        
        # Get top features for humor (positive coefficients)
        humor_features = sorted(zip(feature_names, coefficients), key=lambda x: x[1], reverse=True)[:10]
        
        # Get top features for non-humor (negative coefficients)
        non_humor_features = sorted(zip(feature_names, coefficients), key=lambda x: x[1])[:10]
        
        print("Top 10 features indicating HUMOR:")
        for feature, coef in humor_features:
            print(f"  {feature}: {coef:.4f}")
        
        print("\nTop 10 features indicating NOT HUMOR:")
        for feature, coef in non_humor_features:
            print(f"  {feature}: {coef:.4f}")
            
except Exception as e:
    print(f"Could not analyze features: {e}")

print("\n" + "=" * 60)
print("HUMOR DETECTION MODEL TRAINING COMPLETED!")
print("=" * 60)
print(f"Best Model: {best_model_name}")
print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"Dataset Size: {len(df_sample)} samples")
print("Models trained and evaluated successfully!")
print("Visualizations created and saved")
print("Best model saved for future use")

print("\nNext steps:")
print("- Run 'python gradio_app_simple.py' to launch the web interface")
print("- Use the saved model files for deployment")
print("- Experiment with different preprocessing techniques")
print("- Try ensemble methods for better performance")

print(f"\nModel Performance Summary:")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")