import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

# initialise the flask
app = Flask(__name__)

# Define html file to get user input

@app.route('/')
def home():
    return render_template('main.html')

#step 3

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,9)
    loaded_model = pickle.load(open("dtc.pkl","rb"))
    result= loaded_model.predict(to_predict)
    return result[0]
# output page and logic

@app.route('/result', methods=['POST'])
def result():
    if request.method== 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        return render_template("main.html",result=result)

# main function

if __name__ == "__main__":
    
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"]=True