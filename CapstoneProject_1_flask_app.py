from werkzeug.wrappers import Request, Response
from flask import Flask, request, jsonify
import uuid
import os
import numpy as np
from werkzeug.serving import run_simple
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 입력 data validation
EXPECTED = {
    "cylinders": {"min": 3, "max": 8},
    "displacement": {"min": 68.0, "max": 455.0},
    "horsepower": {"min": 46.0, "max": 230.0},
    "weight": {"min": 1613, "max": 5140},
    "acceleration": {"min": 8.0, "max": 24.8},
    "year": {"min": 70, "max": 82},
    "origin": {"min": 1, "max": 3}
}

# model load
model = load_model(os.path.join("./dnn/", "mpg_model.h5"))


@app.route("/api/mpg/", methods=['POST'])
def calc_mpg():
    content = request.json
    errors = []
    #input validity check
    for name in content:
        if name in EXPECTED:
            value = content[name]
            if value < EXPECTED[name]['min'] or value > EXPECTED[name]['max']:
                error.append(f"입력값 범위 초과 - {name}: {value}")
        else:
            error.append(f"Unexpected field - {name}")
    # missing input check
    for name in EXPECTED:
        if name not in content:
            errors.append(f"missing input - {name}")

    # prediction
    if len(errors) < 1:
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
        response = {"id": str(uuid.uuid4()), "mpg": mpg, "errors": errors}
    else:
        response = {"id": str(uuid.uuid4()), "errors": errors}

    print(response)

    return jsonify(response)


if __name__ == '__main__':
    run_simple('localhost', 9000, app)
#     app.run(host='0.0.0.0', debug=True)
