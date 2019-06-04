# -*- coding: utf-8 -*-
import audio_processing as ap
import os
import models as md
import numpy as np
import random
from keras.models import load_model
from time import time
from keras.callbacks import TensorBoard
from keras import backend as K


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
        ''' Return percent of correct predictions. '''
        counter = 0.0
        for i in range(y_true.shape[0]):
                if(y_true[i] == y_pred[i]):
                        counter = counter + 1
        return (counter*100)/y_true.shape[0]


def create_transcription(prediction, path, window_width=25, verbose=False):
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
                                        print("{}:\t{} sample\t{} sample".format(old, start, stop))

                                fout.write("{},{},{}\n".format(old, start, stop))
                                start = stop = 0
                                old = None
                if verbose == True:
                        print("{}:\t{} sample\t{} sample".format(old, start, stop))

                fout.write("{},{},{}\n".format(old, start, stop))


def train_model(name,
                model,
                model_weights_path,
                test_func,
                epochs,
                alphabet_path,
                dataset_path,
                restore,
                tensorboard=False,
                verbose=False):

        # load alphabet and dataset from given paths
        # dataset = ap.get_dataset(alphabet_path, dataset_path)
        dataset = ap.get_dataset_clarin(alphabet_path, dataset_path)

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
                verbose=False):

        # load alphabet and dataset from given paths
        # dataset = ap.get_dataset(alphabet_path, dataset_path)
        dataset = ap.get_dataset_clarin(alphabet_path, dataset_path)
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

if __name__ == "__main__":
    import argparse

    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--dataset",
        "-d",
        help="path to dataset root directory",
        dest="dataset",
        required=True)
    parser.add_argument("--alphabet",
        "-a",
        help="path to alphabet",
        dest="alphabet",
        required=True)
    parser.add_argument("--weights",
        "-w",
        help="path to weights",
        dest="weights",
        required=True)
    parser.add_argument("--epochs",
        "-e",
        help="number of epochs",
        dest="weights",
        default=100,
        required=False)

    # read arguments from the command line
    args = parser.parse_args()

    model, test_func = md.best_model(38)

    train_model(name="clarin",
        model=model,
        model_weights_path=args.weights,
        test_func=test_func,
        epochs=args.epochs,
        alphabet_path=args.alphabet,
        dataset_path=args.dataset,
        restore=False,
        tensorboard=False,
        verbose=True)