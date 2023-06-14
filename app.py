#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
import unittest

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()

class TestCheckURL(unittest.TestCase):
    def test_1(self):
        resultado = check_url("https://ellibrodepython.com/python-testing")
        self.assertGreater(resultado, 0.8)

    def test_2(self):
        resultado = check_url("https://www.youtube.com/watch?v=N3tJgQK51GQ")
        self.assertGreater(resultado, 0.8)
        

def check_url(url):
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30) 

    y_pred =gbc.predict(x)[0]
    #1 is safe       
    #-1 is unsafe
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    y_pro_non_phishing = gbc.predict_proba(x)[0,1]
    # if(y_pred ==1 ):
    pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
    xx =round(y_pro_non_phishing,2)
    
    return xx


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    # unittest.main()
    app.run(debug=True)