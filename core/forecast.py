# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
from sklearn.externals import joblib

is_normalized = True

def load_scaler(name='test'):
  target_dir = './output/' + name
  scaler = joblib.load(target_dir+"_scaler.save")
  return scaler 

def make_forecast(model,input_x,n_input):
  # reshape into [1, n_input, 1]
  input_x = input_x.reshape((1, len(input_x), 1))
  # forecast the next week
  yhat = model.predict(input_x, verbose=0)
  # print('===== forecast' , n_input)
  # flatten data
  data = np.array(input_x)
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
 
def forecast(data,name,n_input=30,path='./output/'):
  target_dir = path + name
  model = load_model(target_dir + '.model')
  model.load_weights(target_dir + '.weights')
  last_step = data
  pred = make_forecast(model,last_step,n_input)
  
  return pred

if __name__ == "__main__":
  # last 30 days data of the streamHeight
  last_step = np.array([113.08333333,113.08333333, 129.4, 138.13541667, 133.43157895,
    121.24210526, 113.39361702, 114.22340426, 110.96808511, 109.71875,
    108.93548387, 105.17894737, 109.29166667, 111.03125, 104.11702128,
    100.52083333, 87.69473684, 117.36458333, 129.66666667, 92.4742268,
    79.21875, 80.11458333, 85.44791667,  96.13541667, 95.38947368,
    97.54166667, 100.03157895, 102.87368421, 110.04210526, 100.92857143])
  name = 'sr'
  if is_normalized:
    scaler = load_scaler(name)
    last_step = scaler.transform([last_step])
    last_step = last_step[0]
  # construct the argument parse and parse the arguments
  pred = forecast(last_step,name)
  pred = pred[1]
  if is_normalized:
    pred = scaler.inverse_transform([pred])
  
  print(name + ' last step {} forecast to {}'.format(last_step,pred))
