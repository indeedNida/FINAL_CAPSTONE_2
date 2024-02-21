from flask import Flask, request, jsonify, render_template
import re
from collections import defaultdict

app = Flask(__name__)

# Simple N-gram model
class NGramPredictor:
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(lambda: 0))

    def train(self, text):
        words = re.findall(r'\w+', text.lower())
        for i in range(len(words) - self.n):
            self.ngrams[tuple(words[i:i+self.n])][words[i+self.n]] += 1

    def predict(self, text):
        words = re.findall(r'\w+', text.lower())
        if len(words) < self.n:
            return ""
        last_n_words = tuple(words[-self.n:])
        possible_next_words = self.ngrams[last_n_words]
        if not possible_next_words:
            return ""
        return max(possible_next_words, key=possible_next_words.get)

# Initialize the predictor and train with some sample text
predictor = NGramPredictor(n=2)
sample_text = "This is a simple test text for the Next Word Predictor. This test text is for testing."
predictor.train(sample_text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    next_word = predictor.predict(text)
    return jsonify({'next_word': next_word})

if __name__ == '__main__':
    app.run(debug=True)

