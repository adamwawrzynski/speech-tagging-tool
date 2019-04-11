# -*- coding: utf-8 -*-
import audio_processing as ap
import models as md
import numpy as np
from keras.models import load_model
from time import time
from keras.callbacks import TensorBoard
from keras import backend as K


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
                        model.fit(x, y, batch_size=1, epochs=1,
                                        callbacks=[tensorboard],
                                        validation_data=(X_test, y_test))

        model.save(name + ".hd5")
        print("Model saved to disk")


def decode_batch(result):
        ''' Return indexes of the maximum probability for each phoneme
        prediction. '''
        out = result
        ret = np.empty(1)
        for j in range(out.shape[0]):
                out_best = list(np.argmax(out[j,:], axis=1))
                ret = np.append(ret, out_best)
        return ret[1:]


def evaluate_predictions(y_true, y_pred):
        ''' Accumulate wrong predictions. '''
        counter = 0
        for i in range(y_true.shape[0]):
                if(y_true[i] != y_pred[i]):
                        counter = counter + 1
        return counter


if __name__ == "__main__":
        # load alphabet and dataset from given paths
        dataset = ap.get_dataset("../data/phonemes.txt",
                                        "/home/adam/Downloads/TIMIT/TRAIN")

        # load model
        model, test_func = md.custom_ctc_lstm()

        # run tensorboard logger
        tensorboard = TensorBoard(log_dir="logs/" + "custom_ctc_lstm")

        # split dataset to training and testing
        train_test_ratio=0.1
        train_dataset = dataset[int(len(dataset)*train_test_ratio):]
        test_dataset = dataset[0:int(len(dataset)*train_test_ratio)]

        # get one sample of test dataset to validate model
        # model expects input data with shape: [batch_size, timesteps, features]
        # in our case it is 1 element of batch, variable length and 26 MFCC features
        X_test = test_dataset[0].features
        X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        y_test = test_dataset[0].phonemes

        # create list of input lengths
        # CTC loss function expects:
        # input_length with shape: [batch_size, 1], with length of input sequence
        # label_length with shape: [batch_size, 1], with length of output sequence
        input_length = np.ones((1, 1))
        label_length = np.ones((1, 1))

        epochs = 50

        for i in range(0, epochs):

                # dictionary is to store dataset, because of its variable length
                for k in train_dataset:
                        x = k.features
                        y = k.phonemes

                        input_length[0][0] = x.shape[0]
                        label_length[0][0] = y.shape[0]

                        # reshape input data from [None, 26] to [batch_size, None, 26]
                        x = x.reshape(1, x.shape[0], x.shape[1])

                        # reshape output to [batch_size, expected_sequence_length]
                        y = y.reshape(y.shape[0], -1)
                        y = y.reshape(y.shape[1], y.shape[0])

                        # train model
                        # argument x is a tuple of elements necesary for CTC loss function
                        # argument y is dummy variable of shape [expected_sequence_length]
                        model.fit(x=[x, y, input_length, label_length],
                                        y=np.zeros(x.shape[0]),
                                        batch_size=1,
                                        epochs=1,
                                        callbacks=[tensorboard])

                # predict sequence
                # it returns matrix of occurance probability for each phoneme
                result = test_func([X_test])

                # now we choose the biggest probability for each timestep
                out = decode_batch(result[0])
                out = np.asarray(out)

                # print results
                #print("Predicted: \t{}".format(out))
                #print("Actual: \t{}".format(y_test))
                wrong_predictions = evaluate_predictions(y_test, out)
                print("Wrong predictions: {}/{}".format(wrong_predictions, len(out)))