#! /usr/bin/env python
from flask import Flask, render_template, request, jsonify 
from werkzeug.utils import secure_filename
import os

import numpy as np

import keras
from PIL import Image

from keras import backend as K
from skimage import transform
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None
img = None

all_labels = ['Atelectasis',
                'Cardiomegaly',
                'Consolidation',
                'Edema',
                'Effusion',
                'Emphysema',
                'Fibrosis',
                'Hernia', 
                'Infiltration',
                'Mass',
                'Nodule',
                'Pleural_Thickening',
                'Pneumonia',
                'Pneumothorax']

# Loading a keras model with flask
# https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
def load_model(ModelName):
    global model
    global graph
    model = keras.models.load_model(f'Models/{ModelName}.h5')


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# image = load(test_image2)
# results = model.predict(image)
# results

# for label in all_labels:
#     load_model(label)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods = ['GET', 'POST'])
def submit():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)


            image = load(filepath)
            predicted_digit = model.predict(image)

            # Use the model to make a prediction
            data["prediction"] = predicted_digit

            data["prediction"] = [i * 2 for i in data["prediction"]]

            # indicate that the request was a success
            data["success"] = True
            
            print(type(data["prediction"]))

            return render_template('submit.html',dataStuff = data["prediction"])

    return render_template('submit.html')

if __name__ == '__main__':
   load_model("MO")
   model._make_predict_function()
   app.run(debug = True ,port=5555)
