# 📧 Spam Email Classifier

A complete Machine Learning project that classifies email/SMS messages as **Spam** or **Not Spam (Ham)** using a **Multinomial Naive Bayes** classifier with **TF-IDF** text vectorization, wrapped in an interactive **Streamlit** web application.

---

## 📁 Project Structure

```
spam_classifier/
│
├── dataset/
│   └── spam_emails.csv          # Labeled dataset (text, label)
│
├── notebooks/
│   └── spam_classifier_analysis.ipynb   # EDA & model building notebook
│
├── models/
│   ├── model.pkl                 # Trained Naive Bayes model (generated)
│   ├── vectorizer.pkl             # Fitted TF-IDF vectorizer (generated)
│   ├── metrics.pkl                # Evaluation metrics (generated)
│   └── confusion_matrix.png       # Confusion matrix plot (generated)
│
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training script (run this first)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🎯 Objective

Build a machine learning system that automatically distinguishes between **spam** (unwanted/promotional/fraudulent) messages and **ham** (legitimate) messages, and serve predictions through an easy-to-use web interface.

---

## 🛠️ Technologies Used

| Technology     | Purpose                                  |
|-----------------|-------------------------------------------|
| Python          | Core programming language                |
| Pandas          | Data loading and manipulation             |
| NumPy           | Numerical operations                      |
| Scikit-learn    | ML model, TF-IDF, evaluation metrics      |
| NLTK            | Stopword removal & stemming               |
| Matplotlib      | Visualization (confusion matrix, EDA)     |
| Streamlit       | Interactive web application               |
| Pickle          | Model serialization/deserialization       |

---

## ⚙️ Machine Learning Workflow

1. **Data Loading** – Read the labeled dataset (`text`, `label`) from CSV.
2. **Data Cleaning** – Remove nulls/duplicates, encode labels (ham=0, spam=1).
3. **Text Preprocessing** – Lowercasing, URL/punctuation/number removal.
4. **Tokenization** – Split text into word tokens.
5. **Stopword Removal & Stemming** – Remove common words, reduce words to root form.
6. **TF-IDF Vectorization** – Convert text into numerical feature vectors (top 3000 features).
7. **Train-Test Split** – 80% training, 20% testing (stratified).
8. **Model Training** – Multinomial Naive Bayes classifier.
9. **Evaluation** – Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
10. **Model Persistence** – Save model & vectorizer using `pickle`.

---

## 💻 Installation & Setup

### 1. Clone / Download the project
```bash
cd spam_classifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
This generates `model.pkl`, `vectorizer.pkl`, `metrics.pkl`, and `confusion_matrix.png` inside `models/`.
```bash
python train_model.py
```

### 5. Run the Streamlit web app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Using the Web App

1. Enter (or select an example) email/message text in the text box.
2. Click **"🔍 Classify Message"**.
3. View the prediction: **SPAM** 🚨 or **NOT SPAM (HAM)** ✅, along with the confidence percentage.
4. Check the sidebar **Model Dashboard** for accuracy, precision, recall, F1-score, and the confusion matrix.

---

## 📊 Model Performance

Run `train_model.py` to generate live metrics. On the included dataset, the model achieves:

- **Accuracy:** ~95–100%
- **Precision:** High precision on spam detection
- **Recall:** High recall on spam detection
- **F1-Score:** Balanced performance metric

*(Exact values are printed during training and displayed on the Streamlit dashboard.)*

---

## 📓 Notebook

`notebooks/spam_classifier_analysis.ipynb` contains a step-by-step exploratory data analysis and model-building walkthrough, including class distribution plots, message-length analysis, and the full training pipeline.

---

## 🔮 Future Improvements

- Use a larger, real-world dataset (e.g., SMS Spam Collection / Enron Spam dataset).
- Experiment with other algorithms (Logistic Regression, SVM, Random Forest, deep learning).
- Add support for multiple languages.
- Deploy the app on Streamlit Cloud / Heroku / AWS.
- Add email header analysis (sender domain, attachments) for richer features.

---

## 👤 Author

Spam Email Classifier — AI/ML Mini Project
