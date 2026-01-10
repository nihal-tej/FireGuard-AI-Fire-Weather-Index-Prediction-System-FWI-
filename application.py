import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

application = Flask(__name__)
app = application

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
std_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST','GET'])
def predict_datapoint():
  if request.method=="POST":
     Temperature = float(request.form['Temperature'])
     RH = float(request.form['RH'])
     Ws = float(request.form['Ws'])
     Rain = float(request.form['Rain'])
     FFMC = float(request.form['FFMC'])
     DMC = float(request.form['DMC'])
     DC = float(request.form['DC'])
     ISI = float(request.form['ISI'])
     Classes = float(request.form['Classes'])
     Region = float(request.form['Region'])

     columns = [
    'Temperature', 'RH', 'Ws', 'Rain',
    'FFMC', 'DMC','DC', 'ISI', 'Classes', 'Region'
     ]

     new_data = pd.DataFrame([[Temperature, RH, Ws, Rain,
                           FFMC, DMC,DC, ISI, Classes, Region]],
                          columns=columns)
     scaled_data = std_model.transform(new_data)
     result = ridge_model.predict(scaled_data)

     return render_template('home.html', result=result[0])
  return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
