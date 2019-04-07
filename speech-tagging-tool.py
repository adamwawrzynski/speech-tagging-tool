import argparse

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
                    help="defines width of window in milliseconds",
                    dest="frame_width",
                    default='25',
                    required=False)
parser.add_argument("--frame_imposition",
                    "-i",
                    help="defines width of window impositions on both sides in milliseconds",
                    dest="frame_imposition",
                    default='10',
                    required=False)
parser.add_argument("--framing_function",
                    "-f", help="defines framing function",
                    dest="framing_function",
                    default='hamming',
                    required=False)

# read arguments from the command line
args = parser.parse_args()
