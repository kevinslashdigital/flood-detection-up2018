from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
 
# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    # train, test = data[1:-328,0:1], data[-328:-6,0:1]
    train, test = data[0:-322,0:1], data[-322:,0:1]
    print(train.shape)
    print(test.shape)

    # print(train,test)
    # restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
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
def to_supervised(train, n_input, n_out=7):
    # flatten data
    print('to_supervised',train.shape)
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
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
    verbose, epochs, batch_size = 1, 2, 16
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
def forecast(model,input_x):
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    return yhat

def predict(model, history, n_input):
    # print('===== forecast' , n_input)
    # flatten data
    data = array(history)
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
     
# evaluate a single model
def evaluate_model(model,train, test, n_input):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    lst_input = list()
    for i in range(len(test)):
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
        # predict the week
        x ,yhat_sequence = forecast(model, history, n_input)
        print('x',x)
        print('predictions',yhat_sequence)
        # store the predictions
        predictions.append(yhat_sequence)
        lst_input.append(x)
        # print(test[i, :])
        
    # evaluate predictions days for each week
    predictions = array(predictions)
    

    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


if __name__ == "__main__":
    n_input = 7
    # load the new file
    dataset = read_csv('dataset/household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    print('head',dataset.head(15))
    print('tail',dataset.tail(15))
    # split into train and test
    train, test = split_dataset(dataset.values)
    # prepare data & convert history into inputs and outputs
    train_x, train_y = to_supervised(train, n_input)
    test_x, test_y = to_supervised(test, n_input)
    print('test last', test[-1,:])
    # train/fit model
    model = build_model(train_x, train_y, n_input)
    print('==== train head ====')
    print('train.shape',train.shape,test.shape)
    print('==== to_supervised ====')
    print('train.shape',train_x.shape,train_y.shape)
    # test prediction
    predictions = audit(model, test_x,test_y)
    # predictions = array(predictions)
    score, scores = evaluate_forecasts(test_y, predictions)
    print('score',score)
    # forecast to next 7 step
    last_step = test[-1,:]
    pred = forecast(model,last_step)
    print('last step {} forecast to {}'.format(last_step,pred))
    # plot next 7 step forecast
    days = ['2010-11-27', '2010-11-28', '2010-11-29', '2010-11-30', '2010-11-31', '2010-12-01', '2010-12-02']
    pyplot.plot(days, pred[0], marker='o', label='lstm')
    pyplot.show()