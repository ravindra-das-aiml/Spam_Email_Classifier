📧 Spam Email Classifier

A machine learning web app that classifies an email / SMS message as Spam or Not Spam (Ham) using NLP preprocessing, TF-IDF features and a Multinomial Naive Bayes classifier. The app is built with Streamlit and shows the prediction, confidence score and a model performance dashboard.

Developed as part of a 1-month internship at Codec Technologies.

📌 Features
Classifies a message as Spam or Not Spam (Ham)
Shows prediction confidence and probability breakdown
Sidebar model dashboard with Accuracy, Precision, Recall, F1-Score and Confusion Matrix
Built-in example messages for quick testing
Complete NLP pipeline: lowercasing, URL/punctuation/number removal, stopword removal and stemming
Jupyter notebooks for data exploration
Project report and viva Q&A included
🛠️ Tech Stack
Category	Tools
Language	Python 3
Data Handling	Pandas, NumPy
NLP	NLTK (stopwords, Porter Stemmer)
ML	scikit-learn (TF-IDF, Multinomial Naive Bayes)
Visualization	Matplotlib
Web App	Streamlit
📂 Project Structure
Spam_Email_Classifier/
├── .devcontainer/                  # Dev container configuration
├── dataset/
│   └── spam_emails.csv             # Dataset (text, label)
├── models/                         # Saved after training
│   ├── model.pkl                   # Trained Naive Bayes model
│   ├── vectorizer.pkl              # Fitted TF-IDF vectorizer
│   ├── metrics.pkl                 # Evaluation metrics
│   └── confusion_matrix.png        # Confusion matrix plot
├── notebooks/                      # EDA and experiments
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── Project_Report.docx             # Detailed project report
├── Viva_Questions_and_Answers.docx # Viva preparation
└── README.md
⚙️ Methodology
Data Loading: Read the labelled dataset (text, label).
Data Cleaning: Remove missing values and duplicate emails, map ham → 0 and spam → 1.
Text Preprocessing: Lowercase, remove URLs, punctuation and numbers, tokenize, remove stopwords, apply Porter stemming.
Feature Extraction: TF-IDF vectorization (top 3000 features).
Train-Test Split: 80% training / 20% testing (stratified).
Model Training: Multinomial Naive Bayes.
Evaluation: Accuracy, Precision, Recall, F1-Score and Confusion Matrix.
Deployment: Save model with Pickle and serve predictions through Streamlit.
📊 Results
Metric	Score
Accuracy	fill after training
Precision	fill after training
Recall	fill after training
F1-Score	fill after training

Run python train_model.py and copy the printed scores into this table.

🚀 Installation & Usage

1. Clone the repository

bash
git clone https://github.com/ravindra-das-aiml/Spam_Email_Classifier.git
cd Spam_Email_Classifier

2. Install dependencies

bash
pip install -r requirements.txt

3. Train the model (run once)

bash
python train_model.py

4. Run the app

bash
streamlit run app.py

Open http://localhost:8501 in your browser.

Note: NLTK stopwords are downloaded automatically on the first run, so an internet connection is needed once.

🧪 Example
Message	Prediction
Congratulations! You've WON a free iPhone! Call now to claim your prize!	🚨 Spam
Hey, are we still meeting for lunch tomorrow at noon?	✅ Not Spam
📸 Screenshots

Add screenshots of the app here.

![Home](screenshots/home.png)
![Result](screenshots/result.png)
🔮 Future Improvements
Try other models (SVM, Logistic Regression, Random Forest, LSTM, BERT)
Train on a larger and more diverse dataset
Add email header and URL based features
Deploy as a public web app
🎓 Internship

This project was developed during a 1-month internship at Codec Technologies.

👤 Author

Ravindra Das GitHub: @ravindra-das-aiml
