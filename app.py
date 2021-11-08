# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: deepesh
"""

#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.models import load_model

from tensorflow.keras.models import load_model
# Load your trained model
model = load_model('model/model_simple1.h5')
print('@@ Model loaded')

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x).round(3)
    print('@@ Raw result = ', preds)

    preds=np.argmax(preds)
    if preds==0:
        preds="COVID19 AFFECTED PATIENT"
    elif preds==1:
        preds="NORMAL PATIENT"
    else:
        preds="PNEUMONIA AFFECTED PATIENT"
    
    
    return preds


# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        filename = f.filename        
        print("@@ Input posted = ", filename)

        # Save the file to ./uploads
        file_path = os.path.join('static/user uploaded', filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,) 
