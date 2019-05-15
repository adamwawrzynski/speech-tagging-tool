#!/usr/bin/python
import argparse
import numpy as np
import modules.train as train
import modules.models as model
import modules.audio_processing as ap

# initiate the parser
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--source",
                    "-s",
                    help="path to sound file",
                    dest="source",
                    required=True)
parser.add_argument("--dest",
                    "-d",
                    help="path to destination file",
                    dest="destination",
                    required=True)
parser.add_argument("--option",
                    "-o",
                    help="defines whether to return phonemes (-P) or words (-W) tagging",
                    dest="option",
                    default='-P',
                    required=False)
parser.add_argument("--frame_width",
                    "-w",
                    help="defines width of window in seconds",
                    dest="frame_width",
                    default='0.025',
                    type=float,
                    required=False)
parser.add_argument("--frame_imposition",
                    "-i",
                    help="defines width of window impositions on both sides in seconds",
                    dest="frame_imposition",
                    default='0.01',
                    type=float,
                    required=False)
parser.add_argument("--framing_function",
                    "-f", help="defines framing function",
                    dest="framing_function",
                    default='hamming',
                    required=False)

# read arguments from the command line
args = parser.parse_args()

# convert string to function
if 'hamming' in args.framing_function:
    args.framing_function = np.hamming
elif 'hanning' in args.framing_function:
    args.framing_function = np.hanning
elif 'blackman' in args.framing_function:
    args.framing_function = np.blackman
elif 'bartlett' in args.framing_function:
    args.framing_function = np.bartlett
else:
    args.framing_function = np.hamming

# load model
model, test_func = model.custom_ctc_cnn_lstm2_simple()

train.predict_model(name="modules/custom_ctc_cnn_lstm2_simple",
        model=model,
        test_func=test_func,
        audio_path=args.source,
        transcription_path=args.destination,
        alphabet_path="data/phonemes.txt",
        framing_function=args.framing_function,
        frame_width=args.frame_width,
        frame_imposition=args.frame_imposition,
        verbose=True)
