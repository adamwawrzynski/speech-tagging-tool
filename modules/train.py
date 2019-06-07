# -*- coding: utf-8 -*-

__version__ = '1.0'
__author__ = 'Wawrzyński Adam, Szypryt Kamil'

import os
import numpy as np
import random
from keras.models import load_model
from time import time
from keras.callbacks import TensorBoard
from keras import backend as K
import modules.models as md
import modules.audio_processing as ap


def decode_batch(result):
    """Returns indexes of the maximum probability for each phoneme prediction."""

    out = result
    ret = np.empty(1)
    for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j,:], axis=1))
            ret = np.append(ret, out_best)
    return ret[1:]


def evaluate_predictions(y_true, y_pred):
    """Returns percent of correct predictions."""

    counter = 0.0
    for i in range(y_true.shape[0]):
            if(y_true[i] == y_pred[i]):
                    counter = counter + 1
    return (counter*100)/y_true.shape[0]


def create_transcription(prediction, path, window_width=25, verbose=False):
    """Prints start and end sound sample of phoneme occurance.

    Example:\n
    For sound file with 16000 sample per 1 second:\n
    <PHONEME> 'sil'\n
    <START_SAMPLE> 0\n
    <END_SAMPLE> 160000\n

    It meas that phoneme 'sil':
    * starts with 0/16000 second (0 second) of sound file
    * ends in 16000/16000 seconds (1 second) of sound file
    """

    old = None
    start = 0
    stop = 0
    with open(path, "w") as fout:
        if verbose == True:
            print("<PHONEME> <START_SAMPLE> <END_SAMPLE>")

        fout.write("<PHONEME>,<START_SAMPLE>,<END_SAMPLE>\n")
        for i in range(0, len(prediction)):
            if old == prediction[i]:
                stop += window_width

            elif old is None:
                old = prediction[i]
                start = i*window_width
                stop = start

            else:
                if verbose == True:
                    print("{}:\t{} sample\t{} sample".format(old, int(start), int(stop)))

                fout.write("{},{},{}\n".format(old, int(start), int(stop)))
                start = stop = 0
                old = None

        if verbose == True:
                print("{}:\t{} sample\t{} sample".format(old, int(start), int(stop)))

        fout.write("{},{},{}\n".format(old, int(start), int(stop)))


def train_model(name,
                model,
                model_weights_path,
                test_func,
                epochs,
                alphabet_path,
                dataset_path,
                restore,
                language,
                tensorboard=False,
                verbose=False):
    """Trains model and saves pretrained weights to file."""

    # load alphabet and dataset from given paths
    # load alphabet and dataset from given paths
    if language == "polish":
        dataset = ap.get_dataset_clarin(alphabet_path, dataset_path)
    elif language == "english":
        dataset = ap.get_dataset(alphabet_path, dataset_path)
    else:
        print("Lanugage {} is not supported.".format(language))
        exit()

    if not dataset:
        print("Dataset is empty.")
        print("Check your dataset path and selected language.")
        exit()

    # load model to retrain
    if restore == True:
        if os.path.isfile(model_weights_path):
            model.load_weights(model_weights_path)
            print("Model weights loaded from disk")
        else:
            print("Model weights not found")

    # run tensorboard logger
    tb = TensorBoard(log_dir="logs/" + name)

    callbacks = None
    if tensorboard == True:
        callbacks = [tb]

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

    for i in range(0, epochs):

        random.shuffle(train_dataset)

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
                            callbacks=callbacks)

        # predict sequence
        # it returns matrix of occurance probability for each phoneme
        result = test_func([X_test])

        # now we choose the biggest probability for each timestep
        out = decode_batch(result[0])
        out = np.asarray(out)

        # print results
        if verbose == True:
            print("Predicted: \t{}".format(out))
            print("Actual: \t{}".format(y_test))

        predictions = evaluate_predictions(y_test, out)
        print("Correct predictions: {}%".format(round(predictions, 2)))

    model.save_weights(model_weights_path)
    print("Model weights saved to disk")


def evaluate_model(model,
                model_weights_path,
                test_func,
                alphabet_path,
                dataset_path,
                language,
                verbose=False):
    """Checks and prints accuracy of pretrained model on given dataset."""

    # load alphabet and dataset from given paths
    if language == "polish":
        dataset = ap.get_dataset_clarin(alphabet_path, dataset_path)
    elif language == "english":
        dataset = ap.get_dataset(alphabet_path, dataset_path)
    else:
        print("Lanugage {} is not supported.".format(language))
        exit()

    if not dataset:
        print("Dataset is empty.")
        print("Check your dataset path and selected language.")
        exit()

    phonemes = ap.get_feasible_phonemes(alphabet_path)

    # load model to retrain
    if os.path.isfile(model_weights_path):
        model.load_weights(model_weights_path)
        print("Model weights loaded from disk")
    else:
        print("Model weights not found")
        exit()

    accumulate = 0
    for i in range(0, len(dataset)):
        # get one sample of test dataset to validate model
        # model expects input data with shape: [batch_size, timesteps, features]
        # in our case it is 1 element of batch, variable length and 26 MFCC features
        X_test = dataset[i].features
        X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
        y_test = dataset[i].phonemes

        # predict sequence
        # it returns matrix of occurance probability for each phoneme
        result = test_func([X_test])

        # now we choose the biggest probability for each timestep
        out = decode_batch(result[0])
        out = np.asarray(out, dtype=int)

        # print results
        if verbose == True:
            print("Predicted: \t{}".format(out))
            print("Actual: \t{}".format(y_test))

        predictions = evaluate_predictions(y_test, out)
        out = ap.convert_number_to_phoneme(out, phonemes)
        accumulate = accumulate + predictions
    print("Correct predictions: {}%".format(round(accumulate/len(dataset), 2)))


def predict_model(model,
                model_weights_path,
                test_func,
                audio_path,
                transcription_path,
                alphabet_path,
                frame_width=0.025, 
                framing_function=np.hamming,
                frame_imposition=0.01,
                verbose=False):
    """Predicts most probable sequence of phonemes and writes it to file."""

    phonemes = ap.get_feasible_phonemes(alphabet_path)

    # load model weights
    if os.path.isfile(model_weights_path):
        model.load_weights(model_weights_path)
        print("Model weights loaded from disk")
    else:
        print("Model weights not found")
        exit()

    audio, width = ap.process_audio(audio_path, 
                    frame_width=frame_width, 
                    frame_imposition=frame_imposition,
                    framing_function=framing_function)
    audio = audio.reshape(1, audio.shape[0], audio.shape[1])

    # predict sequence
    # it returns matrix of occurance probability for each phoneme
    result = test_func([audio])

    # now we choose the biggest probability for each timestep
    result = decode_batch(result[0])
    result = np.asarray(result, dtype=int)

    result = ap.convert_number_to_phoneme(result, phonemes)
    create_transcription(result, transcription_path, window_width=width, verbose=verbose)
