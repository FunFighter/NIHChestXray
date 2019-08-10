from flask import Flask, render_template, request, jsonify 
from werkzeug.utils import secure_filename
import os

import numpy as np

import keras
from keras.preprocessing import image
from keras import backend as K

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
    model = keras.models.load_model(f'./Models/{ModelName}.h5')




# for label in all_labels:
#     load_model(label)


@app.route('/')
def index():
    return render_template('index.html')

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    # get the request parameters
    params = request.json
    if (params == None):
        params = request.args
    # if parameters are found, echo the msg parameter 
    if (params != None):
        data["response"] = params.get("msg")
        data["success"] = True
    # return a response in json format 
    return jsonify(data)


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

            # Load the saved image using Keras and resize it to the mnist
            # format of 230x230 pixels
            img = image.load_img(filepath, target_size=(230, 230))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            predicted_digit = model.predict(images, batch_size=10)
     
            print(predicted_digit)

            # Get the tensorflow default graph and use it to make predictions
            
            

            # Use the model to make a prediction
            
            data["prediction"] = predicted_digit

            # indicate that the request was a success
            data["success"] = True

            return render_template('submit.html',dataStuff = data["prediction"])

    return render_template('submit.html')

if __name__ == '__main__':
   load_model("Cardiomegaly")
   model._make_predict_function()
   app.run(debug = True ,port=5555)