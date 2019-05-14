from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import scipy.io.wavfile as wav
import sox
import numpy as np
import os


class Sample(object):
    ''' Class representing sound signal with tagged phonemes. '''
    def __init__(self):
        self.features = None
        self.phonemes = None

    def __str__(self):
        print("Features: {}".format(self.features))
        print("Phonemes: {}".format(self.phonemes))

    def set_features(self, features):
        self.features = features

    def set_phonemes(self, phonemes):
        self.phonemes = phonemes


def get_feasible_phonemes(path):
    ''' Get feasible phones from file. '''
    phonemes = {}
    if(os.path.isfile(path)):
        file = open(path)
        counter=0
        for line in file:
            val = line[:-1]
            phonemes[val] = counter
            counter = counter + 1
    return phonemes


def get_phonemes_from_file(path):
    ''' Parse .PHN file and return array of tagged phonemes. '''
    file = open(path)
    values = np.empty((3))
    for line in file:
        val = line.split()
        values = np.append(values, np.asarray(val), axis=0)
    values = values.reshape(int(values.shape[0]/3),3)
    values = values[1:]
    return values


def process_audio(source, 
        n_delta=1, 
        numcep=13, 
        frame_width=0.025, 
        frame_imposition=0.01,
        framing_function=np.hamming):
    '''' Transform sound file and calculate MFCC features. '''
    destination = 'tmp.wav'

    # transform original file
    tfm = sox.Transformer()
    tfm.set_output_format(rate=16000)
    tfm.build(source, destination)
    
    # calulate features
    rate,sig = wav.read(destination)

    os.remove(destination)

    mfcc_feat = mfcc(sig, 
                    rate, 
                    numcep=numcep, 
                    winlen=frame_width,
                    winstep=frame_imposition, 
                    winfunc=framing_function)

    if n_delta >= 1:
        d_mfcc_feat = delta(mfcc_feat, 2)
        
    if n_delta == 2:
        d_mfcc_feat2 = delta(d_mfcc_feat, 2)
        d_mfcc_feat = np.concatenate((d_mfcc_feat, d_mfcc_feat2), axis=1)
        
    complete_mfcc = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
    return complete_mfcc


def convert_phonemes_to_number(dataset, phonemes):
    ''' Replace phonemes with correspoding number from feasible phonemes
    dictionary. '''
    for i in dataset:
            i[2] = phonemes[i[2]]
    return dataset


def convert_number_to_phoneme(prediction, phonemes):
    ''' Replace number of correspoding phoneme with phoneme. '''
    result = []
    for i in range(0, len(prediction)):
        for name in phonemes.items():
            if name[1] == prediction[i]:
                result.append(name[0])
    return result


def get_framing_phonemes(dataset, features):
    ''' Calculate phoneme of each sound signal frame. '''
    tmp_dataset = []
    pointer = 0
    for j in dataset:
        stop = j[1].astype(np.int)
        for k in range(pointer, stop, 160):
            tmp_dataset.append(j[2])
            pointer = pointer + 160
    return tmp_dataset


def get_samples(path, feasible_phonemes):
    ''' Extract dataset from path. '''
    dataset = np.empty(1)

    # extract basename of files and remove duplicates
    filelist = os.listdir(path)
    for i in range(0, len(filelist)):
        filelist[i] = os.path.splitext(os.path.basename(filelist[i]))[0]
    filelist = list(dict.fromkeys(filelist))

    # process files
    for filename in filelist:
        sample = Sample()

        # get list of directories
        files = os.listdir(path)
        files.sort()
        sample_name = "sample.wav"

        # if path points to directory do recursive call
        if(os.path.isdir(path + '/' + filename)):
            sample = get_samples(path + '/' + filename, feasible_phonemes)

        # otherwise process files inside directory
        else:
            features = process_audio(path + '/' + filename + ".WAV")

            sample.set_features(features)
            tmp = get_phonemes_from_file(path + '/' + filename + ".PHN")

            # convert phoneme from ASCII to number representing class
            tmp = convert_phonemes_to_number(tmp, feasible_phonemes)

            # calculate frames of phonemes
            phonemes = np.asarray(get_framing_phonemes(tmp, sample.features),
                                    dtype=int)

            # remove excessive phoneme
            phonemes = phonemes[:features.shape[0]]

            sample.set_phonemes(phonemes)

        # add processed sample to dataset
        dataset = np.append(dataset, sample)
    # remove excessive first row
    dataset = dataset[1:]
    return dataset


def get_dataset(feasible_phonemes_path, dataset_path):
    ''' Remove samples where audio duration don't match transcription duration. '''
    phonemes = get_feasible_phonemes(feasible_phonemes_path)

    dataset = get_samples(dataset_path, phonemes)

    new_dataset = np.empty(1)
    for i in range(0, len(dataset)):
            if dataset[i].features.shape[0] == dataset[i].phonemes.shape[0]:
                    new_dataset = np.append(new_dataset, dataset[i])

    dataset = new_dataset[1:]
    return dataset
