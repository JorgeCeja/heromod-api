from sys import platform as _platform
import json
from flask import Flask, jsonify, render_template, request

from modHero import ModHero

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
app = Flask(__name__)

# load modHero class (model + self contained helper functions)
modHero = ModHero(MODEL_PATH, WEIGHTS_PATH, MAXLEN, LABELS)


def run_inference_on_text(text):
    results = modHero.classify(text)

    return results


@app.route('/v1/api/classify', methods=['POST'])
def classifyText():
    results = run_inference_on_text(request.form['text'])
    response = modHero.postprocessOuput(results)

    return json.dumps({"results": response})


if __name__ == '__main__':
    app.run()
