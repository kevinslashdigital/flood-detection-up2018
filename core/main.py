
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time #helper libraries
from libs import lstm

if __name__ == "__main__":
    #Step 1 Load Data
    X_train, y_train, X_test, y_test = lstm.load_data('dataset/sp500.csv', 50, True)
    print(X_train.shape,y_train.shape)

    model = lstm.build_model((1,50,100,1))

    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=1,
        validation_split=0.05)

    #Step 3 Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=1,
        validation_split=0.05)

    predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
    lstm.plot_results_multiple(predictions, y_test, 50)

    