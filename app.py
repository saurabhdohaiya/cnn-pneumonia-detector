# Importing libraries and modules
# Importing tf and numpy
from flask.helpers import flash
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

# Image generator from keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Importing for loading the saved model
import os.path
from tensorflow.keras.models import load_model

# Importing for making classification and confusion report
# from sklearn.metrics import classification_report, confusion_matrix

# Importing flask and Initializing flask
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer


#Defining the flask app
app = Flask(__name__)


# Load model
model = load_model('model/model.h5')
model.make_predict_function()

# Defining predict function
def make_predict(imgPath, model):
    test_image = image.load_img(imgPath, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    # training_set.class_indices
    if result[0][0] > 0.5:
        return "It's a pneumonia sample"
    else : 
        return "It's a normal sample"
    # return prediction


# app routing
# for heading to home page
@app.route('/', methods=['GET'])
def index():
    # Main Page
    return render_template('index.html')

# for uploading an image
@app.route('/predict', methods=['GET', 'POST'])
def uploadImg():
    if request.method == 'POST' :
        # Getting file from the post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Calliing makePrediction function for performing on the recently uploaded img
        prediction = make_predict(file_path, model)
        return render_template("index.html", data=prediction)
    return None


# main function 
if __name__ == '__main__':
    app.run(debug=True)