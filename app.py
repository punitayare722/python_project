from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data (only run once)
nltk.download("punkt")
nltk.download("stopwords")

# Load trained model and vectorizer
model = joblib.load("sentiment_model.sav")
vectorizer = joblib.load("vectorizer.sav")

# Initialize stemmer
stemmer = PorterStemmer()

# Define negation words
negation_words = set([
    "not", "no", "n't", "dont", "don't", "doesnt", "doesn't", "isnt", "isn't",
    "wasnt", "wasn't", "didnt", "didn't", "wont", "won't", "cant", "can't"
])

# Create custom stopword list by removing negations
default_stopwords = set(stopwords.words('english'))
custom_stopwords = default_stopwords - negation_words

# Function to handle negations
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

# Final preprocessing function (same as training)
def stemming(content):
    if not content:
        return ""

    content = handle_negation(content)
    content = re.sub(r'[^a-zA-Z_\s]', ' ', content)
    content = content.lower().split()

    processed_content = []
    for word in content:
        if word.startswith("not_"):
            processed_content.append(word)
        elif word not in custom_stopwords:
            processed_content.append(stemmer.stem(word))
    return ' '.join(processed_content)

# Home route to handle form submission and show sentiment prediction
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        # Get user input from the form
        user_input = request.form["user_input"]

        # Preprocess the input text
        preprocessed_text = [stemming(user_input)]
        vector = vectorizer.transform(preprocessed_text)

        # Predict sentiment
        prediction = model.predict(vector)[0]

        # Map prediction to label
        label_map = {0: "negative", 1: "positive"}
        sentiment = label_map[prediction]

    return render_template("index.html", sentiment=sentiment)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
