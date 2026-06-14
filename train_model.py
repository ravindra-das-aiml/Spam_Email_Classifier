"""
train_model.py
================
Spam Email Classifier - Model Training Script

This script performs the complete Machine Learning workflow:
1. Data Loading
2. Data Cleaning
3. Text Preprocessing (lowercasing, punctuation/number removal, stopword removal, stemming)
4. Tokenization
5. TF-IDF Vectorization
6. Train-Test Split
7. Model Training using Multinomial Naive Bayes
8. Model Evaluation (Accuracy, Precision, Recall, F1 Score)
9. Confusion Matrix visualization
10. Saving the trained model and vectorizer using Pickle

Run this script once before launching the Streamlit app:
    python train_model.py
"""

import os
import re
import string
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------------------------------------------------
# 0. SETUP - Download required NLTK data (only first time)
# ----------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "spam_emails.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Step 1: Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
print("\nStep 2: Cleaning data...")

# Drop rows with missing values
df = df.dropna(subset=["text", "label"])

# Remove duplicate emails
df = df.drop_duplicates(subset=["text"])

# Standardize label column: map ham -> 0, spam -> 1
df["label"] = df["label"].str.strip().str.lower()
df["target"] = df["label"].map({"ham": 0, "spam": 1})

print(f"Shape after cleaning: {df.shape}")
print(df["label"].value_counts())


# ----------------------------------------------------------------------
# 3. TEXT PREPROCESSING + 4. TOKENIZATION
# ----------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Clean and preprocess raw email text:
      - Convert to lowercase
      - Remove URLs
      - Remove punctuation and numbers
      - Tokenize into words
      - Remove stopwords
      - Apply stemming (Porter Stemmer)

    Returns a cleaned, space-joined string ready for vectorization.
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Tokenization (simple whitespace split - avoids extra nltk punkt dependency issues)
    tokens = text.split()

    # Remove stopwords and short tokens, then apply stemming
    cleaned_tokens = [
        STEMMER.stem(word) for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]

    return " ".join(cleaned_tokens)


print("\nStep 3 & 4: Preprocessing and tokenizing text...")
df["cleaned_text"] = df["text"].apply(preprocess_text)
print(df[["text", "cleaned_text"]].head())

# ----------------------------------------------------------------------
# 5. TF-IDF VECTORIZATION
# ----------------------------------------------------------------------
print("\nStep 5: Applying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["cleaned_text"]).toarray()
y = df["target"].values

print(f"Feature matrix shape: {X.shape}")

# ----------------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ----------------------------------------------------------------------
print("\nStep 6: Splitting data into train and test sets (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 7. MODEL TRAINING (Multinomial Naive Bayes)
# ----------------------------------------------------------------------
print("\nStep 7: Training Multinomial Naive Bayes model...")

model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training complete.")

# ----------------------------------------------------------------------
# 8. MODEL EVALUATION
# ----------------------------------------------------------------------
print("\nStep 8: Evaluating model performance...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ----------------------------------------------------------------------
# 9. CONFUSION MATRIX
# ----------------------------------------------------------------------
print("\nStep 9: Generating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")

# Add labels and ticks
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Ham", "Spam"])
ax.set_yticklabels(["Ham", "Spam"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix - Spam Classifier")

# Annotate cells with counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="black", fontsize=14, fontweight="bold")

fig.colorbar(im, ax=ax)
plt.tight_layout()

cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# ----------------------------------------------------------------------
# 10. SAVE MODEL AND VECTORIZER USING PICKLE
# ----------------------------------------------------------------------
print("\nStep 10: Saving model and vectorizer...")

with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# Save evaluation metrics for display in the Streamlit dashboard
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "confusion_matrix": cm.tolist(),
}

with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

print("\nAll artifacts saved successfully in the 'models/' directory:")
print("  - model.pkl          (trained Naive Bayes model)")
print("  - vectorizer.pkl      (fitted TF-IDF vectorizer)")
print("  - metrics.pkl         (evaluation metrics)")
print("  - confusion_matrix.png (confusion matrix plot)")

print("\nTraining pipeline completed successfully!")
