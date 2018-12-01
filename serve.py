from sys import platform as _platform
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import text, sequence
import json
from flask import Flask, jsonify, render_template, request

graph = tf.get_default_graph()

# model path
PWD = '/var/www/html/flaskapp'
MODEL_PATH = './model/model.json'
# PWD + '/data/model.json'
WEIGHTS_PATH = './model/model.h5'
# PWD + '/data/model.h5'

# preprocessing
MAXLEN = 100

LABELS = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]

# HTTP API
model = None
app = Flask(__name__)


def loadModel():
    global model

    # load json and create model
    json_file = open(MODEL_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(WEIGHTS_PATH)


def preprocessText(textInput):
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts([str(textInput)])
    model_input = tokenizer.texts_to_sequences([str(textInput)])
    model_input = sequence.pad_sequences(model_input, maxlen=MAXLEN)

    return model_input


def postprocessOuput(results):
    responses = []
    for result in results:
        response = []
        for idx, label in enumerate(LABELS):
            # conver to float to prevent - TypeError: Object of type 'float32' is not JSON serializable
            response.append({label: float(result[idx])})
        responses.append(response)

    return responses


def run_inference_on_text(text):
    model_input = preprocessText(text)

    # Required because of a bug in Keras when using tensorflow graph cross threads
    with graph.as_default():
        # evaluate loaded model on inference input
        results = model.predict(model_input)

    return results


@app.route('/v1/api/classify', methods=['POST'])
def classifyText():
    results = run_inference_on_text(request.form['text'])
    response = postprocessOuput(results)

    return json.dumps(response)


if __name__ == '__main__':
    # load keras model and start flask server
    loadModel()
    app.run()
