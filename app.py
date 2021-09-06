import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import func as fu
from keras.models import load_model
import pandas as pd

#from statsmodels.tsa.ar_model import AutoReg
#import math 


app = Flask(__name__)
model_arima = pickle.load(open('model_sarimax.pkl','rb'))  # load sarimax model

scaler = pickle.load(open('scaler.pkl','rb')) # load scaler model

model_rnn = load_model('my_model.hdf5') # load rnn model
data = np.load('data.npy') #load scaled data




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict(): 
    '''
    For rendering results on HTML GUI
    '''
    
    days = int(request.form['days'])
    #final_features = [np.array(int_features)]
    
    try: 
        #res = fu.predict(30, model_arima, model_rnn, data, scaler)
     #   res_sarimax =model_arima.predict(start=366, end=366+days-1)
        result = fu.predict(30, model_arima, model_rnn, data, scaler)
        values = [int(item) for item in result.values]
        dates = result.index.date
       # res_rnn = fu.forecast_rnn(30,model_rnn,data)
       
    except:
        print('some error has occired')
    
    finally:
        return render_template('index.html', beds=values,dates=dates,day=len(result),data=True)
        
    #print(res, file=sys.stderr)
 #   print(res, file=sys.stderr)
  #  print(res, file=sys.stdout)

def forecast_rnn(days,model,data):
    
    test_predictions = []
    n_features = 1
    first_eval_batch = data[-days:]
    current_batch = first_eval_batch.reshape((1, days, n_features))

    for i in range(days):

        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = model.predict(current_batch)[0]

        # store prediction
        test_predictions.append(current_pred) 

        # update batch to now include prediction and drop first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        
    return test_predictions

   

if __name__ == "__main__":
    app.run(debug=True,port=4801)