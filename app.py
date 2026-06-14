"""
app.py
=======
Spam Email Classifier - Streamlit Web Application

This app allows a user to:
  - Enter an email/SMS message
  - Get a prediction (Spam / Not Spam) using a pre-trained Naive Bayes model
  - View the prediction confidence
  - View overall model performance metrics on a dashboard

Run with:
    streamlit run app.py
"""

import os
import re
import string
import pickle

import streamlit as st
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CUSTOM CSS FOR AN ATTRACTIVE UI
# ----------------------------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        color: #1F3A5F;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .result-box-spam {
        background-color: #FDE8E8;
        border: 2px solid #E63946;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .result-box-ham {
        background-color: #E6F4EA;
        border: 2px solid #2E7D32;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .result-text-spam {
        color: #E63946;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .result-text-ham {
        color: #2E7D32;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #333333;
        margin-top: 5px;
    }
    .metric-card {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# NLTK SETUP
# ----------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ----------------------------------------------------------------------
# TEXT PREPROCESSING FUNCTION (must mirror train_model.py exactly)
# ----------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Clean and preprocess raw email text:
      - Lowercase
      - Remove URLs
      - Remove punctuation and numbers
      - Tokenize
      - Remove stopwords
      - Apply Porter stemming
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    cleaned_tokens = [
        STEMMER.stem(word) for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]
    return " ".join(cleaned_tokens)


# ----------------------------------------------------------------------
# LOAD MODEL, VECTORIZER AND METRICS (cached for performance)
# ----------------------------------------------------------------------
MODEL_DIR = "models"

@st.cache_resource
def load_artifacts():
    """Load the trained model, TF-IDF vectorizer, and saved metrics."""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

    return model, vectorizer, metrics


model, vectorizer, metrics = load_artifacts()

# ----------------------------------------------------------------------
# SIDEBAR - MODEL DASHBOARD
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Model Dashboard")

    if metrics is not None:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        st.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")

        st.markdown("---")
        st.markdown(f"**Training samples:** {metrics['train_size']}")
        st.markdown(f"**Testing samples:** {metrics['test_size']}")

        st.markdown("---")
        st.markdown("### Confusion Matrix")
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            # Render confusion matrix directly if image not available
            cm = metrics["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Ham", "Spam"]); ax.set_yticklabels(["Ham", "Spam"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontweight="bold")
            fig.colorbar(im)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Model metrics not found.\nPlease run `train_model.py` first.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses a **Multinomial Naive Bayes** classifier with "
        "**TF-IDF** features to detect spam emails/SMS messages."
    )

# ----------------------------------------------------------------------
# MAIN PAGE HEADER
# ----------------------------------------------------------------------
st.markdown('<div class="main-title">📧 Spam Email Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter an email or message below to check whether it is '
    'Spam or Not Spam (Ham), powered by Machine Learning.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# MAIN INPUT AREA
# ----------------------------------------------------------------------
if model is None or vectorizer is None:
    st.error(
        "🚫 Trained model not found! Please run `python train_model.py` "
        "first to train and save the model before using this app."
    )
else:
    # Example messages for quick testing
    example_choice = st.selectbox(
        "💡 Try an example message (optional):",
        [
            "-- Select an example --",
            "Congratulations! You've WON a free iPhone! Call 09061701461 now to claim your prize!",
            "Hey, are we still meeting for lunch tomorrow at noon?",
            "URGENT! Your account has been suspended. Click here to verify: http://verify-account.net",
            "Thanks for the help with the project yesterday, really appreciated it.",
        ],
    )

    default_text = "" if example_choice == "-- Select an example --" else example_choice

    user_input = st.text_area(
        "✉️ Enter your email/message text here:",
        value=default_text,
        height=150,
        placeholder="Type or paste the email content here...",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_btn = st.button("🔍 Classify Message", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a message to classify.")
        else:
            # 1. Preprocess
            cleaned = preprocess_text(user_input)

            # 2. Vectorize using the saved TF-IDF vectorizer
            vectorized_input = vectorizer.transform([cleaned]).toarray()

            # 3. Predict
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            confidence = max(probabilities) * 100

            # 4. Display result
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-box-spam">
                        <p class="result-text-spam">🚨 SPAM DETECTED</p>
                        <p class="confidence-text">Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box-ham">
                        <p class="result-text-ham">✅ NOT SPAM (HAM)</p>
                        <p class="confidence-text">Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 5. Show probability breakdown
            st.markdown("#### Prediction Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Not Spam (Ham)", f"{probabilities[0]*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with prob_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Spam", f"{probabilities[1]*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("🔧 View Preprocessed Text (Debug Info)"):
                st.write("**Original text:**", user_input)
                st.write("**Cleaned & tokenized text:**", cleaned)

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999;'>Built with ❤️ using Python, "
    "Scikit-Learn & Streamlit | Spam Email Classifier Project</p>",
    unsafe_allow_html=True,
)
