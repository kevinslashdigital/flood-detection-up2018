# https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from models.rnn import RNN_LSTM

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataset = dataset['streamHeight'].values
  print(dataset)
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back)]
    dataX.append(a)
    dataY.append(dataset[i + look_back])
  return np.array(dataX), np.array(dataY)

def transform_to_supervised(df,previous_steps=1, forecast_steps=1,dropnan=True):
  """
  Transforms a DataFrame containing time series data into a DataFrame
  containing data suitable for use as a supervised learning problem.

  Derived from code originally found at 
  https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

  :param df: pandas DataFrame object containing columns of time series values
  :param previous_steps: the number of previous steps that will be included in the
                          output DataFrame corresponding to each input column
  :param forecast_steps: the number of forecast steps that will be included in the
                          output DataFrame corresponding to each input column
  :return Pandas DataFrame containing original columns, renamed <orig_name>(t), as well as
          columns for previous steps, <orig_name>(t-1) ... <orig_name>(t-n) and columns 
          for forecast steps, <orig_name>(t+1) ... <orig_name>(t+n)
  """
  # df = df.iloc[:,2]
  df = df.loc[:, 'streamHeight']
  print(df.head())
  # original column names
  # col_names = df.columns
  col_names = ['streamHeight']

  # list of columns and corresponding names we'll build from 
  # the originals found in the input DataFrame
  cols, names = list(), list()

  # input sequence (t-n, ... t-1)
  for i in range(previous_steps, 0, -1):
    cols.append(df.shift(i))
    names += [('%s(t-%d)' % (col_name, i)) for col_name in col_names]

  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, forecast_steps):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('%s(t)' % col_name) for col_name in col_names]
    else:
      names += [('%s(t+%d)' % (col_name, i)) for col_name in col_names]

  # put all the columns together into a single aggregated DataFrame
  agg = pd.concat(cols, axis=1)
  agg.columns = names

  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  """
  Frame a time series as a supervised learning dataset.
  Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
  Returns:
    Pandas DataFrame of series framed for supervised learning.
  """
  n_vars = 1 # if type(data) is list else data.shape[1]
  df = DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

def plot(df_sr,df_bbt,df_kc,df_ps):
  df_sr.plot(y='streamHeight',title='Siemreap').figure.savefig('output/sr.png')
  df_bbt.plot(y='streamHeight',title='Battambang').figure.savefig('output/btb.png')
  df_kc.plot(y='streamHeight',title='Kampong Chhnang').figure.savefig('output/kc.png')
  df_ps.plot(y='streamHeight',title='Pursat').figure.savefig('output/ps.png')
  plt.show()

def preprocessing():
  df = pd.read_csv("dataset/flood_data.csv", sep = ",",usecols=['created_at','sensor_location','streamHeight'])
  # replace negative numbers in Pandas Data Frame by zero
  num = df._get_numeric_data()
  num[num < 0] = 0
  print(df.head(20))
  # df['updated_at'] = pd.to_datetime(df['updated_at'],infer_datetime_format=True)
  df['created_at'] = pd.to_datetime(df['created_at'],utc=False)
  df.sort_values(by='created_at',ascending=True,inplace=True,na_position='first') # This
  # df = df.cumsum()
 
  df = df.set_index('created_at')


  df_sr = df.query('sensor_location == "Siemreap"')
  df_bbt = df.query('sensor_location == "Battambang"')
  df_kc = df.query('sensor_location == "Kampong Chhnang"')
  df_ps = df.query('sensor_location == "Pursat"')
  print('b4',df_sr.head(20),df_sr.shape,df_sr.isna().sum())   

  df_sr = df_sr.resample('D').mean()
  print(df_sr.head(20),df_sr.shape, df_sr.isna().sum())   

  # plot(df_sr,df_bbt,df_kc,df_ps)
  return df_sr,df_bbt,df_kc,df_ps

# Model support
def train_model(dataset,previous_steps,forecast_steps):
  # look_back = 2
  # trainX, trainY = create_dataset(df_sr, look_back)

  # print(trainX,trainX.shape[0])
  # print('=============')
  # print(trainY,trainY.shape[0])
  # previous_steps = 2
  # forecast_steps = 2

  # dataset = transform_to_supervised(df_sr,previous_steps=previous_steps, forecast_steps=forecast_steps,dropnan=True)
  dataset = dataset.values.astype('float32')
  # Using 60% of data for training, 40% for validation.
  TRAIN_SIZE = 0.60
  train_size = int(len(dataset) * TRAIN_SIZE)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
  print("Number of entries (training set, test set): " + str((len(train), len(test))))

  print('before =============',train.shape)


  # trainX, trainY = dataset
  trainX = train[:,0:previous_steps]
  trainY = train[:,previous_steps:previous_steps+forecast_steps]

  testX = test[:,0:previous_steps]
  testY = test[:,previous_steps:previous_steps+forecast_steps]

  print('=============',previous_steps+forecast_steps)
  print(trainY.shape, trainX.shape)
  print('=============')
  # n_timesteps = forecast_steps
  # n_features = trainX.shape[1]

  n_timesteps = 1
  n_features = trainX.shape[1]
  # reshape input to be [samples, time steps, features]
  trainX = np.reshape(trainX, (trainX.shape[0],previous_steps , trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], previous_steps, testX.shape[1]))

  batch_size = 155 # 14 56 112
  print('***',type(trainX),testY.shape)
 
  model = RNN_LSTM().create_model(batch_size,input_shape=(n_timesteps, n_features),nClasses=testY.shape[1])
  model.summary()
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=50, batch_size=batch_size, verbose=1)

  # predict and score
  train_predict = model.predict(trainX)
  test_predict = model.predict(testX)
  print('prediction',train_predict,testY)
  # Calculate RMSE.
  score = math.sqrt(mean_squared_error(testY, test_predict))
  # print("Training data score: %.2f RMSE" % rmse_train)
  print("Test data score: %.2f RMSE" % score)


  print(train_predict.shape,test_predict.shape)




  # # Start with training predictions.
  # train_predict_plot = np.empty_like(dataset[:,0:1])
  # print('dataset',dataset.shape)
  # print(dataset)
  # print('train_predict_plot',train_predict_plot.shape,train_predict.shape)
  # train_predict_plot[:, :] = np.nan
  # # train_predict_plot[forecast_steps:len(train_predict) + forecast_steps, :] = train_predict
  # train_predict_plot[0:len(train_predict), :] = train_predict
  # print(train_predict_plot.shape)

  

  # # Add test predictions.
  # test_predict_plot = np.empty_like(dataset[:,0:1])
  # print('test_predict_plot',test_predict_plot.shape)
  # test_predict_plot[:, :] = np.nan
  # test_predict_plot[len(train_predict):len(dataset), :] = test_predict
  
  print("test_predict",testX.shape)
  last_step = testX[-1,:,:]
  last_step = np.reshape(last_step, (last_step.shape[0],previous_steps, last_step.shape[1]))
  print("last_step",last_step.shape)
  
  forecast = model.predict(last_step)
  print('last step = {} forecast = {}'.format(last_step,forecast))


  # Create the plot.
  # plt.figure(figsize = (15, 5))
  # plt.plot(dataset, label = "True value")
  # plt.plot(train_predict_plot, label = "Training set prediction")
  # plt.plot(test_predict_plot, label = "Test set prediction")
  # plt.xlabel("Months")
  # plt.ylabel("1000 International Airline Passengers")
  # plt.title("Comparison true vs. predicted training / test")
  # plt.legend()
  # plt.show()




if __name__ == "__main__":
  df_sr,df_bbt,df_kc,df_ps = preprocessing()
  sr = df_sr.loc[:, 'streamHeight']
  bbt = df_bbt.loc[:, 'streamHeight']
  # print('values',sr,sr.shape)
  # print(sr.tail())
  data = series_to_supervised(sr,1,30)
  print(data)
  train_model(data,30,30)
  

