from flask import Flask, render_template, request
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Download NLTK data (only once)
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load("sentiment_model.sav")
vectorizer = joblib.load("vectorizer.sav")

# Stemmer and stopwords
stemmer = PorterStemmer()
negation_words = set(["not", "no", "n't", "dont", "doesn't", "didn't", "can't", "won't"])
custom_stopwords = set(stopwords.words("english")) - negation_words

def handle_negation(text):
    words = word_tokenize(text.lower())
    negation = False
    processed_words = []

    for word in words:
        if word in negation_words:
            negation = True
            continue
        if negation:
            word = "not_" + word
            negation = False
        processed_words.append(word)
    return " ".join(processed_words)

def stemming(content):
    if not content:
        return ""
    content = handle_negation(content)
    content = re.sub(r'[^a-zA-Z_\s]', ' ', content)
    content = content.lower().split()
    processed = []

    for word in content:
        if word.startswith("not_") or word not in custom_stopwords:
            processed.append(stemmer.stem(word))
    return " ".join(processed)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        processed = stemming(user_input)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
