from flask import Flask, render_template, request, jsonify 
from werkzeug.utils import secure_filename
import os

import keras
from keras.preprocessing import image
from keras import backend as K


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None


# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model(ModelName):
    global model
    global graph
    model = keras.models.load_model(f'{ModelName}.h5')
    graph = K.get_session().graph


# load_model()

def prepare_image(img):
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Scale from 0 to 255
    img /= 255
    # Flatten the image to an array of pixels
    image_array = img.flatten().reshape(-1, 230 * 230)
    # Return the processed feature array
    return image_array


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

            # Load the saved image using Keras and resize it to the mnist
            # format of 230x230 pixels
            image_size = (230, 230)
            im = image.load_img(filepath, target_size=image_size,
                                grayscale=True)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(im)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            # global graph
            # with graph.as_default():

            #     # Use the model to make a prediction
            #     predicted_digit = model.predict_classes(image_array)[0]
            #     data["prediction"] = str(predicted_digit)

            #     # indicate that the request was a success
            #     data["success"] = True

            return '''<div>test  testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest</div> '''
    return render_template('submit.html')

if __name__ == '__main__':
   app.run(debug = True)