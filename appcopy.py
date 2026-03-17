from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data only if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load("sentiment.pkl")
vectorizer = joblib.load("vectorizers.pkl")

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

# Final preprocessing function
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

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    cleaned_text = None
    confidence = None
    error = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "")

        if not user_input.strip():
            error = "Please enter some text."
        else:
            # Preprocess
            cleaned_text = stemming(user_input)
            vector = vectorizer.transform([cleaned_text])

            # Prediction
            prediction = model.predict(vector)[0]

            # ✅ 3-class label mapping
            label_map = {
                0: "negative",
                1: "neutral",
                2: "positive"
            }
            sentiment = label_map.get(prediction, "unknown")

            # ✅ Confidence score
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vector)[0]
                confidence = round(max(prob) * 100, 2)

            # ✅ OPTIONAL: fallback neutral (if binary model)
            elif hasattr(model, "decision_function"):
                score = model.decision_function(vector)[0]
                if abs(score) < 0.5:
                    sentiment = "neutral"

    return render_template(
        "indexs.html",
        sentiment=sentiment,
        cleaned_text=cleaned_text,
        confidence=confidence,
        error=error
    )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)