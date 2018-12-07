# https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
import pandas as pd
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

def plot(df_sr,df_bbt,df_kc,df_ps):
  df_sr.plot(x='created_at',y='streamHeight',title='Siemreap').figure.savefig('output/sr.png')
  df_bbt.plot(x='created_at',y='streamHeight',title='Battambang').figure.savefig('output/btb.png')
  df_kc.plot(x='created_at',y='streamHeight',title='Kampong Chhnang').figure.savefig('output/kc.png')
  df_ps.plot(x='created_at',y='streamHeight',title='Pursat').figure.savefig('output/ps.png')
  plt.show()

# Model support
def train_model():
  df = pd.read_csv("dataset/flood_data.csv", sep = ",",usecols=['created_at','sensor_location','streamHeight'])
  # df['updated_at'] = pd.to_datetime(df['updated_at'],infer_datetime_format=True)

  df['created_at'] = pd.to_datetime(df['created_at'],utc=False)
  df.sort_values(by='created_at',ascending=True,inplace=True,na_position='first') # This
  # df = df.cumsum()
  print(df.head(20))

  df.set_index('created_at')

  df_sr = df.query('sensor_location == "Siemreap"')
  df_bbt = df.query('sensor_location == "Battambang"')
  df_kc = df.query('sensor_location == "Kampong Chhnang"')
  df_ps = df.query('sensor_location == "Pursat"')
  print(df_sr.head(20))

  # look_back = 2
  # trainX, trainY = create_dataset(df_sr, look_back)

  # print(trainX,trainX.shape[0])
  # print('=============')
  # print(trainY,trainY.shape[0])
  previous_steps = 2
  forecast_steps = 1

  dataset = transform_to_supervised(df_sr,previous_steps=previous_steps, forecast_steps=forecast_steps,dropnan=True)
  dataset = dataset.values.astype('float32')
  # Using 60% of data for training, 40% for validation.
  TRAIN_SIZE = 0.60
  train_size = int(len(dataset) * TRAIN_SIZE)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
  print("Number of entries (training set, test set): " + str((len(train), len(test))))
  print(train)

  # trainX, trainY = dataset
  trainX = train[:,0:previous_steps]
  trainY = train[:,previous_steps:previous_steps+forecast_steps]

  testX = test[:,0:previous_steps]
  testY = test[:,previous_steps:previous_steps+forecast_steps]

  print(trainX,trainY,trainX.shape)
  print('=============')
  # reshape input to be [samples, time steps, features]
  # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  trainX = np.reshape(trainX, (1, trainX.shape[0], trainX.shape[1]))
  testX = np.reshape(testX, (1, testX.shape[0], testX.shape[1]))
  
  trainY = np.reshape(trainY, (1, trainY.shape[0], trainY.shape[1]))
  testY = np.reshape(testY, (1, testY.shape[0], testY.shape[1]))

  batch_size = 56 # 14 56 112
  print(type(trainX),trainX)
  model = RNN_LSTM().create_model(batch_size,input_shape=(None,forecast_steps),nClasses=forecast_steps)
  model.summary()
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=10, batch_size=batch_size, verbose=1)


  # predict and score
  train_predict = model.predict(trainX)
  test_predict = model.predict(testX)
  print('prediction',train_predict,testY)
  # Calculate RMSE.
  # score = math.sqrt(mean_squared_error(testY, test_predict))
  # print("Test data score: %.2f RMSE" % score)

  print(testX,testX.shape)

  future_pred_count = 10
  future = []
  currentStep = testX[:,-1:,:] #last step from the previous prediction

  print(currentStep,currentStep.shape)
  for i in range(future_pred_count):
    currentStep = model.predict(currentStep) #get the next step
    future.append(currentStep) #store the future steps    

  print('future',future)

  # Start with training predictions.
  train_predict_plot = np.empty_like(dataset)
  train_predict_plot[:, :] = np.nan
  train_predict_plot[forecast_steps:len(train_predict) + forecast_steps, :] = train_predict

  # Add test predictions.
  test_predict_plot = np.empty_like(dataset)
  test_predict_plot[:, :] = np.nan
  test_predict_plot[len(train_predict) + (forecast_steps * 2) + 1:len(dataset) - 1, :] = test_predict

  # Create the plot.
  plt.figure(figsize = (15, 5))
  plt.plot(dataset, label = "True value")
  plt.plot(train_predict_plot, label = "Training set prediction")
  plt.plot(test_predict_plot, label = "Test set prediction")
  plt.xlabel("Months")
  plt.ylabel("1000 International Airline Passengers")
  plt.title("Comparison true vs. predicted training / test")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  # df = pd.DataFrame( 
  #   {
  #     'Symbol':['A','A','A'] ,
  #     'Date':['02/20/2015','01/15/2016','08/21/2015']
  #   })
  # print(df)

  # df['Date'] =pd.to_datetime(df.Date)
  # df.sort_values(by='Date',inplace=True, ascending=True)
  # print(df)

  # construct the argument parse and parse the arguments
  train_model()
