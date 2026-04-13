"""
Simplified Humor Detection NLP Model
====================================
A streamlined version focusing on core functionality with traditional ML models.

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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

print("Starting Humor Detection Model Training...")


print("Downloading NLTK data...")
try:
    nltk.data.find('tokenizers/punkt')
    print("SUCCESS Punkt tokenizer already available")
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
    print("SUCCESS Stopwords already available")
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

print("\n" + "=" * 60)
print("HUMOR DETECTION NLP MODEL - SIMPLIFIED VERSION")
print("=" * 60)


print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

try:
    df = pd.read_csv('dataset.csv')
    print(f"SUCCESS Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"\nClass distribution:")
    print(df['humor'].value_counts())
    print(f"Humor percentage: {df['humor'].mean():.2%}")
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"ERROR Error loading dataset: {e}")
    exit(1)


print("\n2. DATA CLEANING AND PREPROCESSING")
print("-" * 40)

def clean_text(text):
    """Clean text by removing URLs, punctuation, converting to lowercase, and removing stopwords."""
    try:
        
        text = str(text)
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_text)
    except:
        return ""

print("Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Remove empty texts after cleaning
original_size = len(df)
df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
print(f"SUCCESS Text cleaning completed")
print(f"Removed {original_size - len(df)} empty texts after cleaning")
print(f"Final dataset shape: {df.shape}")

print("\nSample of cleaned text:")
for i in range(min(3, len(df))):
    print(f"Original: {df['text'].iloc[i][:100]}...")
    print(f"Cleaned:  {df['cleaned_text'].iloc[i][:100]}...")
    print()

# Step 3: Train-Test Split
print("\n3. TRAIN-TEST SPLIT (80/20)")
print("-" * 40)

X = df['cleaned_text']
y = df['humor'].astype(int)  # Convert boolean to int

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"SUCCESS Data split completed")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training humor percentage: {y_train.mean():.2%}")
print(f"Test humor percentage: {y_test.mean():.2%}")

# Step 4: Feature Extraction - TF-IDF
print("\n4. FEATURE EXTRACTION - TF-IDF")
print("-" * 40)

print("Extracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Reduced for faster processing
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"SUCCESS TF-IDF extraction completed")
print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# Step 5: Model Training
print("\n5. TRAINING MACHINE LEARNING MODELS")
print("-" * 40)

models = {}
results = {}

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
models['Logistic Regression'] = lr_model
print("SUCCESS Logistic Regression trained")

# SVM (with linear kernel for faster training)
print("Training SVM...")
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train_tfidf, y_train)
models['SVM'] = svm_model
print("SUCCESS SVM trained")

# Step 6: Model Evaluation
print("\n6. EVALUATING MODELS")
print("-" * 40)

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }

# Evaluate all models
for name, model in models.items():
    results[name] = evaluate_model(model, X_test_tfidf, y_test, name)

# Step 7: Model Comparison
print("\n7. MODEL COMPARISON")
print("-" * 40)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1_score'] for model in results.keys()]
})

print(comparison_df.round(4))

best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
print(f"\nBest performing model: {best_model_name}")
print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")

# Step 8: Visualizations
print("\n8. CREATING VISUALIZATIONS")
print("-" * 40)

try:
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Model Performance Comparison
    plt.subplot(2, 3, 1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, comparison_df[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix for Best Model
    plt.subplot(2, 3, 2)
    best_predictions = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Humor', 'Humor'],
                yticklabels=['Not Humor', 'Humor'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 3. Class Distribution
    plt.subplot(2, 3, 3)
    class_counts = df['humor'].value_counts()
    plt.pie(class_counts.values, labels=['Not Humor', 'Humor'], autopct='%1.1f%%', 
            colors=['lightcoral', 'lightblue'])
    plt.title('Class Distribution in Dataset')
    
    # 4. WordCloud for Humorous Text
    plt.subplot(2, 3, 4)
    humor_text = ' '.join(df[df['humor'] == True]['cleaned_text'])
    if humor_text:
        wordcloud_humor = WordCloud(width=400, height=300, background_color='white').generate(humor_text)
        plt.imshow(wordcloud_humor, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud - Humorous Text')
    
    # 5. WordCloud for Non-Humorous Text
    plt.subplot(2, 3, 5)
    non_humor_text = ' '.join(df[df['humor'] == False]['cleaned_text'])
    if non_humor_text:
        wordcloud_non_humor = WordCloud(width=400, height=300, background_color='white').generate(non_humor_text)
        plt.imshow(wordcloud_non_humor, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud - Non-Humorous Text')
    
    # 6. Text Length Distribution
    plt.subplot(2, 3, 6)
    df['text_length'] = df['cleaned_text'].str.len()
    plt.hist(df[df['humor'] == True]['text_length'], alpha=0.7, label='Humor', bins=30)
    plt.hist(df[df['humor'] == False]['text_length'], alpha=0.7, label='Not Humor', bins=30)
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('humor_detection_results.png', dpi=300, bbox_inches='tight')
    print("SUCCESS Visualizations saved as 'humor_detection_results.png'")
    
except Exception as e:
    print(f"WARNING Warning: Could not create all visualizations: {e}")

# Step 9: Save Best Model
print("\n9. SAVING BEST MODEL")
print("-" * 40)

try:
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_humor_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    print(f"SUCCESS {best_model_name} model saved as 'best_humor_model.pkl'")
    print("SUCCESS TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
except Exception as e:
    print(f"ERROR Error saving model: {e}")

# Step 10: Sample Predictions
print("\n10. SAMPLE PREDICTIONS")
print("-" * 40)

sample_texts = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "The stock market crashed today due to economic uncertainty.",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "Climate change is a serious global issue that requires immediate attention.",
    "Why did the scarecrow win an award? He was outstanding in his field!"
]

print("Sample Predictions:")
print("-" * 20)

best_model = models[best_model_name]
for text in sample_texts:
    try:
        cleaned = clean_text(text)
        if cleaned:
            text_tfidf = tfidf_vectorizer.transform([cleaned])
            prediction = best_model.predict(text_tfidf)[0]
            probability = best_model.predict_proba(text_tfidf)[0]
            
            humor_label = "HUMOR" if prediction == 1 else "NOT HUMOR"
            confidence = max(probability)
            print(f"Text: {text}")
            print(f"Prediction: {humor_label} (Confidence: {confidence:.3f})")
            print()
    except Exception as e:
        print(f"Error predicting for text: {text[:50]}... - {e}")

print("\n" + "="*60)
print("HUMOR DETECTION MODEL TRAINING COMPLETED!")
print("="*60)
print(f"SUCCESS Best Model: {best_model_name}")
print(f"SUCCESS Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
print("SUCCESS Models trained and evaluated successfully!")
print("SUCCESS Visualizations created and saved")
print("SUCCESS Best model saved for future use")
print("="*60)

print("\nNext steps:")
print("1. Run 'python gradio_app.py' to launch the interactive web interface")
print("2. Open your browser to http://localhost:7860 to test the model")
print("3. Try different texts to see how well the model detects humor!")