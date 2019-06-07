#!/usr/bin/python
import argparse
import numpy as np
import modules.train as train
import modules.models as model
import modules.audio_processing as ap
import os


def CheckExt(choices):
    """Checks if file has valid extension and return argparse.Action object."""

    class Act(argparse.Action):
        def __call__(self,parser,namespace,fname,option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("file doesn't end with one of {}{}".format(choices,option_string))
            else:
                setattr(namespace,self.dest,fname)

    return Act


# initiate the parser
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--source",
                    "-s",
                    help="path to sound file",
                    dest="source",
                    required=True,
                    action=CheckExt({"wav","WAV"}))
parser.add_argument("--dest",
                    "-d",
                    help="path to destination file",
                    dest="destination",
                    required=True,
                    action=CheckExt({"csv"}))
parser.add_argument("--option",
                    "-o",
                    help="defines whether to return phonemes (-P) or words (-W) tagging",
                    dest="option",
                    default="-P",
                    required=False)
parser.add_argument("--frame_width",
                    "-w",
                    help="defines width of window in seconds",
                    dest="frame_width",
                    default="0.025",
                    type=float,
                    required=False)
parser.add_argument("--frame_imposition",
                    "-i",
                    help="defines width of window impositions on both sides in seconds",
                    dest="frame_imposition",
                    default="0.01",
                    type=float,
                    required=False)
parser.add_argument("--framing_function",
                    "-f", help="defines framing function",
                    dest="framing_function",
                    default="hamming",
                    required=False)
parser.add_argument("--lanugage",
                    "-l",
                    help="defines language of speech",
                    dest="language",
                    default="polish",
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

if args.language == "polish":

    # load model
    model, test_func = model.best_model(38)

    train.predict_model(model=model,
            model_weights_path="modules/polish.weights",
            test_func=test_func,
            audio_path=args.source,
            transcription_path=args.destination,
            alphabet_path="data/phonemes_clarin.txt",
            framing_function=args.framing_function,
            frame_width=args.frame_width,
            frame_imposition=args.frame_imposition,
            verbose=True)

elif args.language == "english":

    # load model
    model, test_func = model.best_model(39)

    train.predict_model(model=model,
            model_weights_path="modules/english.weights",
            test_func=test_func,
            audio_path=args.source,
            transcription_path=args.destination,
            alphabet_path="data/phonemes_timit.txt",
            framing_function=args.framing_function,
            frame_width=args.frame_width,
            frame_imposition=args.frame_imposition,
            verbose=True)

else:
    print("Language: {} is not supported.".format(args.language))
    exit()