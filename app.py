from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Data Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))  # Cache stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Train and save the model
def train_and_save_model():
    # Load dataset
    df = pd.read_csv('dataset.csv')  # Replace with your dataset
    df = df.sample(frac=0.1, random_state=42)  # Use 10% of the data for faster training

    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Feature extraction using TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))  # Reduce features
    X = tfidf.fit_transform(df['cleaned_text'])  # Sparse matrix
    y = df['label'].values

    # Train the model (use SGDClassifier for faster training)
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, n_jobs=-1)  # Correct loss parameter
    model.fit(X, y)

    # Save the model and vectorizer
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved successfully!")

# Load the model and vectorizer
try:
    model = joblib.load('fake_news_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Model files not found. Training the model...")
    train_and_save_model()
    model = joblib.load('fake_news_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')

# Flask App
app = Flask(__name__)

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    cleaned_text = preprocess_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    prediction_proba = model.predict_proba(vectorized_text)[:, 1]
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    })

if __name__ == '__main__':
    app.run(debug=True)