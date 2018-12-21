from math import sqrt
from numpy import split
from numpy import array
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
# from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import os
import math

verbose, epochs, batch_size = 1, 10, 16
time_step = 30
is_normalized = True

def save_scaler(scaler,name='test'):
    target_dir = './output/' + name
    scaler_filename = target_dir + "_scaler.save"
    joblib.dump(scaler, scaler_filename) 

def load_scaler(name='test'):
    target_dir = './output/' + name
    scaler = joblib.load(target_dir+"_scaler.save")
    return scaler 

def plot(data,name):
  data.plot(x='created_at',y='streamHeight',title=name).figure.savefig('output/'+name+'.png')

# split a univariate dataset into train/test sets
def split_dataset(data,name='test'):
    print('split_dataset',data.shape)
    # find number of record can be train
    m = data.shape[0] / time_step
    m = math.floor(m)
    pos_n = int(m * time_step)
    data = data[-pos_n:] #data[-330:]
    data = data.reshape(pos_n,1)
    print('split_dataset',data.shape)
    # normallization
    if is_normalized:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        save_scaler(scaler,name)
    # split into standard weeks
    # train, test = data[1:-328,0:1], data[-328:-6,0:1]
    train, test = data[0:-60,0:1], data[-60:,0:1]
    print('split_dataset train',train.shape)
    print('split_dataset test',test.shape)

    # print(train,test)
    # restructure into windows of weekly data
    train = array(split(train, len(train)/30))
    test = array(split(test, len(test)/30))
    # print(train,test)

    # validate train data
    print(train.shape)
    print(train[0, 0, 0], train[-1,:, 0])
    # validate test
    print(test.shape)
    print(test[0, 0, 0], test[-1, :, 0])
    return train, test
 
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    predicted = array(predicted)
    print('evaluate_forecasts',predicted.shape)
    predicted = predicted.reshape(predicted.shape[0],predicted.shape[2])
    print(actual.shape,predicted.shape)
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores
 
# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))
 
# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=30):
    # flatten data
    print('to_supervised',train.shape)
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    print('len(data)',len(data))
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            # print('in_start {} in_end {} out_end {}'.format(in_start,in_end,out_end))
            x_input = data[in_start:in_end, 0]
            # print(x_input.shape,x_input)
            x_input = x_input.reshape((len(x_input), 1))
            # print(x_input.shape,x_input)
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
            # print('y ',data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)
 
# train the model
def build_model(train_x, train_y, n_input):
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model
 
# make a forecast
def forecast(model,input_x,n_input):
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # return yhat

    # print('===== forecast' , n_input)
    # flatten data
    data = array(input_x)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return input_x,yhat
 
def audit(model,x,y):
    predictions = list()
    for i in range(len(x)):
        input_x = x[i]
        input_x = input_x.reshape((1, len(input_x), 1))
        yhat = model.predict(input_x, verbose=0)
        predictions.append(yhat)
    return predictions

def preprocessing(path="dataset/flood_data.csv"):
    df = pd.read_csv(path, sep = ",",usecols=['created_at','sensor_location','streamHeight'])
    # replace negative numbers in Pandas Data Frame by zero
    num = df._get_numeric_data()
    num[num < 0] = np.nan
    print('original from csv',df.head(20))
    # df['updated_at'] = pd.to_datetime(df['updated_at'],infer_datetime_format=True)
    df['created_at'] = pd.to_datetime(df['created_at'],utc=False)
    df.sort_values(by='created_at',ascending=True,inplace=True,na_position='first') # This
    # df = df.cumsum()
    
    df = df.set_index('created_at')


    df_sr = df.query('sensor_location == "Siemreap"')
    df_bbt = df.query('sensor_location == "Battambang"')
    df_kc = df.query('sensor_location == "Kampong Chhnang"')
    df_ps = df.query('sensor_location == "Pursat"')

    # print('tail sr',df_sr.tail(30))

    # resample to day 'SR'
    df_sr = df_sr.resample('D').mean()
    df_sr = df_sr.fillna(method='ffill')
    # resample to day 'SR'
    df_bbt = df_bbt.resample('D').mean()
    df_bbt = df_bbt.fillna(method='ffill')
    # resample to day 'SR'
    df_kc = df_kc.resample('D').mean()
    df_kc = df_kc.fillna(method='ffill')
    # resample to day 'SR'
    df_ps = df_ps.resample('D').mean()
    df_ps = df_ps.fillna(method='ffill')
    # plot(df_sr,df_bbt,df_kc,df_ps)
    return df_sr,df_bbt,df_kc,df_ps

def save_model(model,name):
    target_dir = './output/' 
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    print("[INFO] serializing network to ", target_dir)
    model.save(target_dir + name + '.model')
    model.save_weights(target_dir + name +'.weights')

def fit_and_save_model(dataset,name):
    print('fit_and_save_model',dataset.tail(30))
    n_input = 30
    n_output = 30
    # split into train and test
    train, test = split_dataset(dataset.values,name)
    # prepare data & convert history into inputs and outputs
    train_x, train_y = to_supervised(train, n_input,n_output)
    test_x, test_y = to_supervised(test, n_input,n_output)
    print('test last', test[-1,:],test.shape)
    print('==== train head ====')
    print('train.shape',train.shape,test.shape)
    print('==== to_supervised ====')
    print('trainxy.shape',train_x.shape,train_y.shape)
    print('testxy.shape',test_x.shape,test_y.shape)
     # train/fit model
    model = build_model(train_x, train_y, n_input)
    # test prediction
    predictions = audit(model, test_x,test_y)
    print('predictions',predictions)
    # predictions = array(predictions)
    score, scores = evaluate_forecasts(test_y, predictions)
    print('test score',score,scores)
    # forecast to next 30 step
    last_step = test[-1,:]
    pred = forecast(model,last_step,n_input)
    pred = pred[1]
    if is_normalized:
        scaler = load_scaler(name)
        pred = scaler.inverse_transform([pred])
    print('{} last step {} forecast to {}'.format(name,last_step,pred))
    # save model
    save_model(model,name)
    # plot next 30 step forecast
    # days = ['2010-11-27', '2010-11-28', '2010-11-29', '2010-11-30', '2010-11-31', '2010-12-01', '2010-12-02']
    # pyplot.plot(days, pred[0], marker='o', label='lstm')
    # pyplot.show()

if __name__ == "__main__":
    # preprocessing
    df_sr,df_bbt,df_kc,df_ps = preprocessing()
    sr = df_sr.loc[:, 'streamHeight']
    bbt = df_bbt.loc[:, 'streamHeight']
    kc = df_kc.loc[:, 'streamHeight']
    ps = df_ps.loc[:, 'streamHeight']

    # print(sr.tail(30))
    # plot(sr,'siem_reab')
    # plot(bbt,'bat_dom_bong')
    # plot(kc,'kompong_cham')
    plot(ps,'posat')

    fit_and_save_model(sr,'sr')
    fit_and_save_model(bbt,'bt')
    fit_and_save_model(kc,'kc')
    fit_and_save_model(ps,'ps')
    