
import pandas as pd
from keras.models import load_model
import numpy as np
import cv2
import os
import librosa
import AudioFeature
import flask
from flask import Flask,render_template,request
App=Flask(__name__)
model=load_model('my_1model.h5')
Imodel = load_model('CNNModel.h5')


                 
@App.route('/')
def Home():
    return render_template('index.html')
@App.route('/TestAudio',methods=['GET','POST'])
def Test():
    if(request.method=='POST'):
        print("#############")
        path=request.files['file']
        print("#############")
        print(path.filename)
        
        
        data,sr=librosa.load(path,duration=2.5,offset=0.6)
        aud=AudioFeature.extract_features(data,sr,2048,512)
        aud=aud.reshape(1,-1)
        print(aud.shape)

        res=model.predict(aud)
        res = (res >= 0.5).astype(int)
        if res==0:
            result='You are a healthy person'
        else:
            result='You are a Parkinson\'s person'
        return render_template('index.html',res=result)
    else:
        return "Error"
@App.route('/TestImage',methods=['GET','POST'])
def testImage():
     if(request.method=='POST'):
        print("#############")
        path=request.files['file']
        print("#############")
        print(path.filename)
        path.save('TestedImage'+path.filename)
        img = cv2.imread('TestedImage'+path.filename)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)

        # Perform prediction
        prediction = Imodel.predict(img_array)
        res = np.argmax(prediction)
        if res==0:
            result='You are a healthy person'
        else:
            result='You are a Parkinson\'s person'
        return render_template('index.html',res=result)

     return 'hi'
@App.route('/record',methods = ['GET', 'POST'])
def Record():
    return render_template('speechRecord.html')
@App.route('/upload',methods = ['GET', 'POST'])
def upload():
    return render_template('speechupload.html')
App.run(debug=True) 