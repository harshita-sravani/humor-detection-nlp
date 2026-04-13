"""
Humor Detection NLP Model
=========================
This script builds and evaluates multiple models for humor detection using:
- Traditional ML: Logistic Regression, SVM with TF-IDF features
- Deep Learning: Fine-tuned DistilBERT

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

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Deep Learning libraries
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("=" * 60)
print("HUMOR DETECTION NLP MODEL")
print("=" * 60)

# Step 1: Load and Explore Data
print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

# Load the dataset
df = pd.read_csv('dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nClass distribution:")
print(df['humor'].value_counts())
print(f"Humor percentage: {df['humor'].mean():.2%}")

# Step 2: Data Cleaning and Preprocessing
print("\n2. DATA CLEANING AND PREPROCESSING")
print("-" * 40)

def clean_text(text):
    """
    Clean text by removing URLs, punctuation, converting to lowercase,
    and removing stopwords.
    """
    # Convert to string if not already
    text = str(text)
    
    # Remove URLs
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

# Apply text cleaning
print("Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)

print("Sample of cleaned text:")
for i in range(3):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Cleaned:  {df['cleaned_text'].iloc[i]}")
    print()

# Remove empty texts after cleaning
df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
print(f"Dataset shape after cleaning: {df.shape}")

# Step 3: Train-Test Split
print("\n3. TRAIN-TEST SPLIT (80/20)")
print("-" * 40)

X = df['cleaned_text']
y = df['humor'].astype(int)  # Convert boolean to int

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training humor percentage: {y_train.mean():.2%}")
print(f"Test humor percentage: {y_test.mean():.2%}")

# Step 4: Feature Extraction - TF-IDF
print("\n4. FEATURE EXTRACTION - TF-IDF")
print("-" * 40)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

print("Extracting TF-IDF features...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# Step 5: Traditional ML Models Training
print("\n5. TRAINING TRADITIONAL ML MODELS")
print("-" * 40)

models = {}
results = {}

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
models['Logistic Regression'] = lr_model

# SVM
print("Training SVM...")
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_tfidf, y_train)
models['SVM'] = svm_model

# Evaluate traditional models
print("\n6. EVALUATING TRADITIONAL ML MODELS")
print("-" * 40)

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
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

# Evaluate traditional models
for name, model in models.items():
    if name in ['Logistic Regression', 'SVM']:
        results[name] = evaluate_model(model, X_test_tfidf, y_test, name)

print("\n7. PREPARING DISTILBERT MODEL")
print("-" * 40)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DistilBERT Dataset class
class HumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize DistilBERT tokenizer and model
print("Loading DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
).to(device)

# Create datasets
train_dataset = HumorDataset(X_train, y_train, tokenizer)
test_dataset = HumorDataset(X_test, y_test, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./humor_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Custom trainer for evaluation metrics
class HumorTrainer(Trainer):
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# Initialize trainer
trainer = HumorTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda eval_pred: HumorTrainer().compute_metrics(eval_pred)
)

print("\n8. TRAINING DISTILBERT MODEL")
print("-" * 40)
print("Training DistilBERT (this may take a while)...")

# Train the model
trainer.train()

# Evaluate DistilBERT
print("\n9. EVALUATING DISTILBERT MODEL")
print("-" * 40)

# Get predictions
predictions = trainer.predict(test_dataset)
y_pred_bert = np.argmax(predictions.predictions, axis=1)

# Calculate metrics
bert_accuracy = accuracy_score(y_test, y_pred_bert)
bert_precision = precision_score(y_test, y_pred_bert)
bert_recall = recall_score(y_test, y_pred_bert)
bert_f1 = f1_score(y_test, y_pred_bert)

print(f"\nDistilBERT Results:")
print(f"Accuracy:  {bert_accuracy:.4f}")
print(f"Precision: {bert_precision:.4f}")
print(f"Recall:    {bert_recall:.4f}")
print(f"F1-Score:  {bert_f1:.4f}")

results['DistilBERT'] = {
    'accuracy': bert_accuracy,
    'precision': bert_precision,
    'recall': bert_recall,
    'f1_score': bert_f1,
    'predictions': y_pred_bert
}

print("\n10. MODEL COMPARISON")
print("-" * 40)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1_score'] for model in results.keys()]
})

print(comparison_df.round(4))

# Find best model
best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
print(f"\nBest performing model: {best_model_name}")

print("\n11. CREATING VISUALIZATIONS")
print("-" * 40)

# Set up the plotting style
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

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
wordcloud_humor = WordCloud(width=400, height=300, background_color='white').generate(humor_text)
plt.imshow(wordcloud_humor, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Humorous Text')

# 5. WordCloud for Non-Humorous Text
plt.subplot(2, 3, 5)
non_humor_text = ' '.join(df[df['humor'] == False]['cleaned_text'])
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
plt.show()

print("Visualizations saved as 'humor_detection_results.png'")

print("\n12. SAVING BEST MODEL")
print("-" * 40)

# Save the best model (DistilBERT)
if best_model_name == 'DistilBERT':
    model.save_pretrained('./best_humor_model')
    tokenizer.save_pretrained('./best_humor_model')
    print("DistilBERT model saved to './best_humor_model'")
else:
    # Save traditional ML model
    import joblib
    if best_model_name == 'Logistic Regression':
        joblib.dump(lr_model, 'best_humor_model.pkl')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    elif best_model_name == 'SVM':
        joblib.dump(svm_model, 'best_humor_model.pkl')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    print(f"{best_model_name} model saved as 'best_humor_model.pkl'")

print("\n13. SAMPLE PREDICTIONS")
print("-" * 40)

# Sample predictions
sample_texts = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "The stock market crashed today due to economic uncertainty.",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "Climate change is a serious global issue that requires immediate attention.",
    "Why did the scarecrow win an award? He was outstanding in his field!"
]

print("Sample Predictions:")
print("-" * 20)

if best_model_name == 'DistilBERT':
    # DistilBERT predictions
    for text in sample_texts:
        # Clean the text
        cleaned = clean_text(text)
        
        # Tokenize
        inputs = tokenizer(cleaned, return_tensors='pt', truncation=True, 
                          padding='max_length', max_length=128).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(prediction, dim=-1).item()
            confidence = prediction[0][predicted_class].item()
        
        humor_label = "HUMOR" if predicted_class == 1 else "NOT HUMOR"
        print(f"Text: {text}")
        print(f"Prediction: {humor_label} (Confidence: {confidence:.3f})")
        print()
else:
    # Traditional ML predictions
    best_traditional_model = models[best_model_name]
    for text in sample_texts:
        cleaned = clean_text(text)
        text_tfidf = tfidf_vectorizer.transform([cleaned])
        prediction = best_traditional_model.predict(text_tfidf)[0]
        probability = best_traditional_model.predict_proba(text_tfidf)[0]
        
        humor_label = "HUMOR" if prediction == 1 else "NOT HUMOR"
        confidence = max(probability)
        print(f"Text: {text}")
        print(f"Prediction: {humor_label} (Confidence: {confidence:.3f})")
        print()

print("\n" + "="*60)
print("HUMOR DETECTION MODEL TRAINING COMPLETED!")
print("="*60)
print(f"Best Model: {best_model_name}")
print(f"Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
print("All models trained and evaluated successfully!")
print("Visualizations and model saved.")
print("="*60)