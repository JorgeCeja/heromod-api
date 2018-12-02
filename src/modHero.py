import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import text, sequence


class ModHero(object):
    def __init__(self, model_path, weights_path, max_len, labels):
        self.model_path = model_path
        self.weights_path = weights_path
        self.max_len = max_len
        self.labels = labels
        self.graph = tf.get_default_graph()
        self.model = self.load_model(self.model_path, self.weights_path)

    def load_model(self, model_path, weights_path):
        # load json and create model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights(weights_path)

        return model

    def classify(self, text):
        model_input = self.preprocessText(text)

        # Required because of a bug in Keras when using tensorflow graph cross threads
        with self.graph.as_default():
            # evaluate loaded model on inference input
            results = self.model.predict(model_input)

        return results

    def preprocessText(self, textInput):
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts([str(textInput)])
        model_input = tokenizer.texts_to_sequences([str(textInput)])
        model_input = sequence.pad_sequences(model_input, maxlen=self.max_len)

        return model_input

    def postprocessOuput(self, results):
        responses = []
        for result in results:
            response = []
            for idx, label in enumerate(self.labels):
                # conver to float to prevent - TypeError: Object of type 'float32' is not JSON serializable
                response.append({label: float(result[idx])})
            responses.append(response)

        return responses
