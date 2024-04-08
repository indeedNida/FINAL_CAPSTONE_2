from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Declare the model, tokenizer, and other parameters as global variables
model = None
tokenizer = None
max_sequence_len = 17  # Update this with the value used during training


def load_model_and_tokenizer():
    global model, tokenizer  # Declare both model and tokenizer as global inside the function
    if model is None:
        model = tf.saved_model.load('trained_model')

    if tokenizer is None:
        with open('dataset/sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
            text = file.read()

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
    return model, tokenizer

# Load the model and tokenizer before running the Flask app
load_model_and_tokenizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.form['seed_text']
    next_words = int(request.form['next_words'])
    num_results = 3  # Number of result sets to generate

    # Temperature for diversity (adjust as needed)
    temperature = 0.6

    # Generate the text
    results = []

    for _ in range(num_results):
        predicted_words = []
        current_seed = seed_text

        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([current_seed])[0]

            # Ensure that the token_list has the same length as max_sequence_len
            token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

            # Adjusted prediction step using the serving signature
            serving_fn = model.signatures['serving_default']
            token_tensor = tf.convert_to_tensor(token_list, dtype=tf.int32)
            predictions = serving_fn(embedding_input=tf.cast(token_tensor, tf.float32))[list(serving_fn.structured_outputs.keys())[0]]

            # Apply temperature scaling
            predictions = predictions.numpy()[0]
            predictions = np.log(predictions) / temperature
            exp_predictions = np.exp(predictions)
            predicted_probabilities = exp_predictions / np.sum(exp_predictions)

            # Sample the next word based on the modified probabilities
            predicted_index = np.random.choice(len(predicted_probabilities), p=predicted_probabilities)
            output_word = ""

            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break

            current_seed += " " + output_word
            predicted_words.append(output_word)

        predicted_text = " ".join(predicted_words)
        results.append(predicted_text)

    return jsonify({'results': results})


if __name__ == "__main__":
    app.run()
