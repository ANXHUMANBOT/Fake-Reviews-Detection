from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np # Import numpy

# Initialize Flask app
app = Flask(__name__)

# --- Load Model & Preprocessing Tools ---
try:
    model = joblib.load('fake_review_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    print("--- ERROR ---")
    print("Model or vectorizer .joblib files not found.")
    print("Please make sure 'fake_review_model.joblib' and 'tfidf_vectorizer.joblib' are in the same folder as app.py")
    exit()

# Download NLTK resources (stopwords)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    """Cleans and prepares a single review text for prediction."""
    if not isinstance(text, str):
        return ""
        
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Keep only letters
    text = text.lower()                     # Convert to lowercase
    words = text.split()                    # Split into words
    # Stem and remove stopwords
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)
# ----------------------------------------

# Define the main page route
@app.route('/')
def home():
    # This will send your index.html file to the browser
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    try:
        # Get the text from the POST request
        review_text = request.json['text']

        # 1. Preprocess the text
        processed_text = preprocess_text(review_text)
        
        if not processed_text.strip():
             return jsonify({'prediction': 'N/A', 'confidence': 0, 'message': 'Please enter more text.'})

        # 2. Vectorize the text
        text_vector = vectorizer.transform([processed_text])

        # 3. Make prediction
        # --- FIX: Use predict_proba for robustness instead of model.predict() ---
        probability = model.predict_proba(text_vector)

        # model.classes_ is [0, 1] so probability[0] is [P(Genuine), P(Fake)]
        prob_genuine = probability[0][0]
        prob_fake = probability[0][1]

        if prob_fake > prob_genuine:
            result = 'Fake'
            confidence = prob_fake
        else:
            result = 'Genuine'
            confidence = prob_genuine
        # --- END FIX ---

        # 4. Return the result as JSON
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence*100:.2f}"
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True) # Run the app in debug mode

