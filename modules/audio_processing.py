from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import scipy.io.wavfile as wav
import sox
import numpy as np

def get_features(source, destination, n_delta=1, numcep=13):
    '''' Transform original sound file and calculate features. '''
    # transform original file
    tfm = sox.Transformer()
    tfm.set_output_format(rate=16000)
    tfm.build(source, destination)

    # calulate features
    rate,sig = wav.read(destination)
    mfcc_feat = mfcc(sig, rate, numcep=numcep)
    fbank_feat = logfbank(sig, rate)
    if n_delta >= 1:
        d_mfcc_feat = delta(mfcc_feat, 2)
    if n_delta == 2:
        d_mfcc_feat2 = delta(d_mfcc_feat, 2)
        d_mfcc_feat = np.concatenate((d_mfcc_feat, d_mfcc_feat2), axis=1)
    complete_mfcc = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
    return complete_mfcc
