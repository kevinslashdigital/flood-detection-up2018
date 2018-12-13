# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2

def make_forecast(model,input_x,n_input):
  # reshape into [1, n_input, 1]
  input_x = input_x.reshape((1, len(input_x), 1))
  # forecast the next week
  yhat = model.predict(input_x, verbose=0)
  # return yhat

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
 
def forecast(data,name,n_input=30):
  target_dir = './ouput/' + name
  model = load_model(target_dir + '.model')
  model.load_weights(target_dir + '.weights')
  last_step = data
  pred = make_forecast(model,last_step,n_input)
  print(name + ' last step {} forecast to {}'.format(last_step,pred))
  return pred

if __name__ == "__main__":
  # last 30 days data of the streamHeight
  data = np.array([342.62162162,342.62162162,342.3625,422,523.47311828,497.06451613,422.98958333,
        264.07291667, 382.75, 374.26041667,189.94791667,283.54736842,244.98958333,
        99.25263158, 64.10416667, 79.48421053,31.91489362,88.10416667,146.12765957,
        104.82105263,112.375, 111.81052632,29.47368421,59.8 ,95.93684211,192.15625,
        96.5 ,36.73333333, 54.96875   ,83.25 ])
  # construct the argument parse and parse the arguments
  forecast(data,'ps')
