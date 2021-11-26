from flask import Flask, request, jsonify
import os
import numpy as np
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = MobileNet(weights='imagenet', include_top=True)

@app.route('/api/image', methods=['POST'])
def upload_image():
  # check if the post request has the file part
  if 'image' not in request.files:
      return jsonify({'error': 'No posted image. Should be attribute named image.'})
  file = request.files['image']

  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
      return jsonify({'error': 'Empty filename submitted.'})
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      print("***2**:"+filename)
      x = []
      ImageFile.LOAD_TRUNCATED_IMAGES = False
      img = Image.open(BytesIO(file.read()))
      img.load()
      img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      pred = model.predict(x)
      lst = decode_predictions(pred, top=5)

      items = []
      for itm in lst[0]:
        items.append({'name': itm[1], 'prob': float(itm[2])})

      response = {'pred': items}
      return jsonify(response)
  else:
      return jsonify({'error': 'File has invalid extension'})


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
