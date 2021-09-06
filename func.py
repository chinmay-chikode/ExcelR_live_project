# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:23:32 2021

@author: Chinmay
"""
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import math 
import pandas as pd

def predict(days,model_arima,model_rnn,data,scaler):
    
    forecast_Sarimax = model_arima.predict(start=366, end=366+days-1).rename('SARIMAX Predictions')
   # forecast_Rnn = scaler.inverse_transform(forecast_rnn(days,model_rnn,data))
    #true_rnn_predictions = scaler.inverse_transform(rnn_predictions)

    #final_forecast = (forecast_Sarimax + forecast_Rnn[:,0])/2

    
    df_temp = pd.read_csv('beds.csv')

    tmp = df_temp['Available Beds'].append(forecast_Sarimax)

    tmp = pd.DataFrame(tmp.values,columns=['Available Beds'])

    series=tmp.iloc[:,0]
    # create lagged dataset
    values = pd.DataFrame(series.values)
    dataframe = pd.concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t', 't+1']

    X = dataframe.values

    train_size = 366
    train_res, test_res = X[1:train_size], X[train_size:]
    train_X, train_y = train_res[:,0], train_res[:,1]
    test_X, test_y = test_res[:,0], test_res[:,1]

    # calculate residuals
    train_resid = [train_y[i]-train_X[i] for i in range(len(train_X))] #difference between both columns

    window = 15
    model = AutoReg(train_resid, lags=15)
    model_fit = model.fit()
    coef = model_fit.params


    # walk forward over time steps in test
    history = train_resid[-window:]
    # history = [history[i] for i in range(len(history))]
    predictions_res = []
    for t in range(len(test_X)):
        # persistence
        yhat = test_X[t]  # actual data or data at time 't'
        error = test_y[t] - yhat  #difference between time at 't' and 't+1'
        # predict error
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        pred_error = coef[0]
        for d in range(window):
            pred_error += coef[d+1] * lag[window-d-1]
        # correct the prediction
        yhat = yhat + pred_error
        predictions_res.append(yhat)
        history.append(error)
        
        
    result_forecast = forecast_Sarimax+history[-days:]
    
    for i in range(len(result_forecast)):
        result_forecast[i] = int(np.round(result_forecast[i]))
        '''
        if ((num % math.floor(num)) > 0.175) and ((num % math.floor(num)) <0.5):
            result_forecast[i] = int(np.round(num + 0.326))
        else:
            result_forecast[i] = (np.round(num))
            '''

    return result_forecast
    #return 5



def forecast_rnn(days,model,data):
    
    test_predictions = []
    n_features = 1
    first_eval_batch = data[-days:]
    current_batch = first_eval_batch.reshape((1, days, n_features))
    current_pred = model.predict(current_batch)[0]
    '''
    for i in range(days):

        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = model.predict(current_batch)[0]

        # store prediction
        test_predictions.append(current_pred) 

        # update batch to now include prediction and drop first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        
    return test_predictions'''
    return current_pred

