from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD, Adam
import numpy as np
from utils import *

def main():
    X_train, Y_train, X_test, Y_test = load_data()
    data_dim = X_train.shape[-1]
    timesteps = X_train.shape[1]
    num_classes = 6
    print("types", X_train.dtype, Y_train.dtype, X_test.dtype, Y_test.dtype)
    model = Sequential()
    model.add(LSTM(256, return_sequences = True, input_shape = (timesteps, data_dim)))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(256))
    model.add(Dense(6, activation='softmax'))
    adam = Adam(lr = 0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer = adam,
                  metrics=['mae'])

    model.fit(X_train, Y_train, batch_size = 8, epochs = 10)
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1])) 


if __name__ == '__main__':
    main()
