# Deploy simple Keras model with Flask only.
from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load neural network model
model = load_model(os.path.join("./dnn/", "mpg_model.h5"))

@app.route('/api/mpg', methods=['POST'])
def calc_mpg():
    content = request.json

    # Predict
    x = np.zeros((1, 7))

    x[0, 0] = content['cylinders']
    x[0, 1] = content['displacement']
    x[0, 2] = content['horsepower']
    x[0, 3] = content['weight']
    x[0, 4] = content['acceleration']
    x[0, 5] = content['year']
    x[0, 6] = content['origin']

    pred = model.predict(x)
    mpg = float(pred[0])
    response = {"mpg": mpg}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='localhost', debug=True)
