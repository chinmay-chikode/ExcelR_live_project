import pandas as pd
import numpy as np
import math

'''import seaborn as sns
import matplotlib.pyplot as plt'''

import warnings
warnings.filterwarnings("ignore")


from statsmodels.tsa.statespace.sarimax import SARIMAX
#from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
#import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
#from datetime import datetime,time
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pickle
from keras.models import load_model
import marshal,types

df = pd.read_csv('Beds_Occupied - Beds_Occupied.csv',parse_dates=True)

df['collection_date'] = pd.to_datetime(df['collection_date'],format='%d-%m-%Y')

df.index = df['collection_date']

df = df.asfreq('d')

df.drop('collection_date',inplace=True,axis=1)

null_data = df[df.isnull().any(axis=1)]

span = 14
alpha = 2/(span+1)

df['EWMA'] = df['Total Inpatient Beds'].ewm(alpha=alpha,adjust=False).mean()

null_data = df[df.isnull().any(axis=1)]

df['Total Inpatient Beds'] = df['Total Inpatient Beds'].fillna(round(df.EWMA))
df['Available Beds'] = 900 - df['Total Inpatient Beds']

df.drop(['Total Inpatient Beds','EWMA'],inplace=True,axis=1)

df.to_csv('beds.csv')

##################################### SARIMAX #####################################################

model_sarimax = SARIMAX(df['Available Beds'],order=(1,1,1))

results_sarimax = model_sarimax.fit()

##################################### RNN #########################################################

scaler= MinMaxScaler()

new_train = df[['Available Beds']]

scaled_train = scaler.fit_transform(new_train)


n_input = 30
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

model_rnn = Sequential()
model_rnn.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model_rnn.add(Dense(36, activation='relu'))
model_rnn.add(Dense(18, activation='relu'))
model_rnn.add(Dense(9, activation='relu'))
model_rnn.add(Dense(1))
model_rnn.compile(optimizer='adam', loss='mse')

model_rnn.fit_generator(generator,epochs=14)


#####################################################################################################
'''
def forecast_rnn(days,model,data):
    test_predictions = []

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
'''
#####################################################################################################
'''
def predict(days):
    forecast_Sarimax = model_arima.predict(start=366, end=366+days-1).rename('SARIMAX Predictions')
    forecast_Rnn = scaler.inverse_transform(func_rnn_forecast(days,model_rnn,data))
    #true_rnn_predictions = scaler.inverse_transform(rnn_predictions)

    final_forecast = (forecast_Sarimax + forecast_Rnn[:,0])/2

    
    df_temp = pd.read_csv('beds.csv')

    tmp = df_temp['Total Inpatient Beds'].append(final_forecast)

    tmp = pd.DataFrame(tmp.values,columns=['Total Inpatient Beds'])

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
        
        
    result_forecast = final_forecast+history[-days:]
    for i in range(len(result_forecast)):
        num = result_forecast[i]
        if ((num % math.floor(num)) > 0.175) and ((num % math.floor(num)) <0.5):
            result_forecast[i] = np.round(num + 0.326)
        else:
            result_forecast[i] = np.round(num)

    return np.round(result_forecast)
'''

#####################################################################################################

pickle.dump(results_sarimax, open('model_sarimax.pkl','wb')) #stored sarimax model

pickle.dump(scaler, open('scaler.pkl', 'wb')) # store scaler model

model_rnn.save('my_model.hdf5')  # save rnn model

data = np.asarray(scaled_train)

np.save('data.npy', data)  # save to npy file

#rnn_pred_code = marshal.dumps(forecast_rnn.__code__) #save rnn function

#predict_code = marshal.dumps(predict.__code__) #save predict function

#########################################################################################################

model_arima = pickle.load(open('model_sarimax.pkl','rb'))  # load sarimax model

scaler = pickle.load(open('scaler.pkl','rb')) # load scaler model

model_rnn = load_model('my_model.hdf5') # load rnn model

data = np.load('data.npy') #load scaled data
'''
rnn_pred_func = marshal.loads(rnn_pred_code)
func_rnn_forecast = types.FunctionType(rnn_pred_func, globals(), "some_func_name") # load function for rnn

predict_code = marshal.loads(predict_code)
func_predict_code = types.FunctionType(predict_code, globals(), "some_func_name") # load function for prediction
'''

#######################################################################################################

days=30
#result_data = func_predict_code(days)

res = fu.predict(30,model_arima,model_rnn,data,scaler)
