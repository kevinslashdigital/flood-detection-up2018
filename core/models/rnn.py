from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class RNN_LSTM():
  def __init__(self):
    pass

  def create_model(self,batch_size,input_shape,nClasses):
    
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=input_shape))
    # model.add(LSTM(32,batch_input_shape=(batch_size, input_shape[1], input_shape[2]), return_sequences=False,stateful=True))
    model.add(Dense(nClasses))
    
    return model
