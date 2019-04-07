import audio_processing as ap
import models as md
import numpy as np
from keras.models import load_model
import os
from time import time
from keras.callbacks import TensorBoard


def train_model(model, name, dataset, epochs=25, train_test_ratio=0.1, restore=False):
        tensorboard = TensorBoard(log_dir="logs/" + name)

        # split dataset to training and testing
        train_dataset = dataset[int(len(dataset)*train_test_ratio):]
        test_dataset = dataset[0:int(len(dataset)*train_test_ratio)]

        # get one sample of test dataset to validate model
        X_test = test_dataset[0].features
        X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        y_test = test_dataset[0].phonemes
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])

        # load model to retrain
        if restore == True:
                model = load_model(name + ".hd5")
                print("Model loaded from disk")

        for i in range(0, epochs):
                for k in train_dataset:
                        x = k.features
                        x = x.reshape(1, x.shape[0], x.shape[1])
                        y = k.phonemes
                        y = y.reshape(1, y.shape[0], y.shape[1])
                        model.fit(x, y, batch_size=1, epochs=1, callbacks=[tensorboard], validation_data=(X_test, y_test))

        model.save(name + ".hd5")
        print("Model saved to disk")

if __name__ == "__main__":
        dataset = ap.get_dataset("../data/phonemes.txt", "/home/adam/Downloads/TIMIT/TRAIN/DR1/FCJF0")

        #model = md.unidirectional_lstm()
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        #train_model(model, "unidirectional_lstm", dataset, epochs=25)

        #model = md.new_cnn_lstm()
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        #train_model(model, "new_cnn_lstm", dataset, epochs=25)

        #model = md.bigger_lstm2()
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        #train_model(model, "bigger_lstm2", dataset, epochs=25)

        model = md.ctc_unidirectional_lstm()
        model.compile(optimizer='rmsprop')

        tensorboard = TensorBoard(log_dir="logs/" + "ctc_unidirectional_lstm")

        # split dataset to training and testing
        train_test_ratio=0.1
        train_dataset = dataset[int(len(dataset)*train_test_ratio):]
        test_dataset = dataset[0:int(len(dataset)*train_test_ratio)]

        nb_train = 1
        nb_test = 1

        # get one sample of test dataset to validate model
        X_test = test_dataset[0].features
        X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        y_test = test_dataset[0].phonemes
        y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])

        # load model to retrain
        restore = False
        if restore == True:
                model = load_model("ctc_unidirectional_lstm" + ".hd5")
                print("Model loaded from disk")
        epochs = 25

        for i in range(0, epochs):
                for k in train_dataset:
                        x = k.features
                        y = k.phonemes

                        # create list of input lengths
                        x_train_len = np.zeros((1, 1)) #np.asarray([len(x[0]) for i in range(0, len(x))]) #
                        y_train_len = np.zeros((1, 1)) #np.asarray([len(y[0]) for i in range(0, len(x))]) #
                        print(x.shape[0])
                        print(y.shape[0])
                        print(x_train_len.shape)
                        print(y_train_len.shape)
                        for i in range(0, 1):
                            x_train_len[i][0] = 172
                        
                        for i in range(0, 1):
                            y_train_len[i][0] = 172

                        # x_train_len = x
                        # y_train_len = y
                        x = x.reshape(1, x.shape[0], x.shape[1])
                        y = y.reshape(1, y.shape[0] * y.shape[1])
                        y = y[:,0:61]
                        print("x.shape {}".format(x.shape))
                        print("y.shape {}".format(y.shape))
                        print("x_train_len.shape {}".format(x_train_len.shape))
                        print("y_train_len.shape {}".format(y_train_len.shape))

                        model.fit(x=[x, y, x_train_len, y_train_len], y=y, batch_size=1) #, epochs=1, callbacks=[tensorboard], validation_data=(X_test, y_test))

        model.save("ctc_unidirectional_lstm" + ".hd5")
        print("Model saved to disk")


