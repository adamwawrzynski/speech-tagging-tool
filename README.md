# speech-tagging-tool

## About

This repository contains tool for labelling phonemes with start and stop
timestamps for English and Polish speech.

* We acquired ~75% accuracy on TIMIT dataset after 15 epochs of training
* We acquired ~80% accuracy on CLARIN-EMU dataset after 15 epochs of training[*](#notes)

## Quick links

* [Requirements](#requirements)
* [Getting started](#getting-started)
* [Architecture](#architecture)
* [Dataset](#dataset)
* [Future development](#future-development)
* [Notes](#notes)

## Requirements

Program was tested on Windows 10 and Ubuntu 18.04 LTS.

To run the **speech-tagging-tool.py** you need Python3.x (recommended version is
3.6) and modules listed in **requirements.txt** file.

## Getting started

### Installation (Ubuntu)

To install all dependencies you have to run the following commands:

```bash
# install Python3.x nad pip3
sudo apt-get install -y python3 python3-dev python3-pip

# install all dependencies
pip3 install -r requirements.txt
```

Or you can run *setup.sh* script with the following command:

```bash
sudo ./setup.sh
```

### Usage of tool

Program, thanks to **argparse** module, offers simple manual and parameters
validation. You will menu if you give wrong parameters or run program with
**-h** flag:

```bash
./speech-tagging-tool.py -h
```

And in result you will see this:

```bash
Using TensorFlow backend.
usage: speech-tagging-tool.py [-h] --source SOURCE --dest DESTINATION
                              [--option OPTION] [--frame_width FRAME_WIDTH]
                              [--frame_imposition FRAME_IMPOSITION]
                              [--framing_function FRAMING_FUNCTION]
                              [--lanugage LANGUAGE]

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE, -s SOURCE
                        path to sound file
  --dest DESTINATION, -d DESTINATION
                        path to destination file
  --option OPTION, -o OPTION
                        defines whether to return phonemes (-P) or words (-W)
                        tagging
  --frame_width FRAME_WIDTH, -w FRAME_WIDTH
                        defines width of window in seconds
  --frame_imposition FRAME_IMPOSITION, -i FRAME_IMPOSITION
                        defines width of window impositions on both sides in
                        seconds
  --framing_function FRAMING_FUNCTION, -f FRAMING_FUNCTION
                        defines framing function
  --lanugage LANGUAGE, -l LANGUAGE
                        defines language of speech
```

## Architecture

[Based on paper: Framewise Phoneme Classification with
Bidirectional LSTM and Other Neural Network
Architectures](https://www.cs.toronto.edu/~graves/nn_2005.pdf)

System is based on recurrent neural network with connectionist temporal
classification (CTC) loss function. As an input networks requires list of 26
features (MFCC + deltas) obtained from audio file. Length of this list can vary.
As a result, network returns sequence of the most probable phonemes for each
frame window.

Next step is to accumulate consecutive phonemes to calculate phoneme boundaries.
As frame window depends on audio file sample rate and window width of framing
function and width of overlay it have to by dynamically computed.

## Dataset

Training for English was performed on TIMIT dataset, which you can download from
here: https://goo.gl/l0sPwz

Training for Polish was performed on CLARIN-EMU dataset, which you can download
from here: http://mowa.clarin-pl.eu/korpusy/clarin_emu.zip

## Future development

We hope that this tool will help scientists to develop large vocabulary speech
recognition systems (LVSR).

### Ideas

We wanted to implement the following features:
* predict words on sequences of phonemes to fix wrong predicted phonemes and
	improve accuracy of model
* predict words on sequences of phonemes to label words with start and stop
	timestamps more accurately

## Notes

Because of resources constraints we couldn't train model on whole CLARIN-EMU
dataset, therefore accuracy is unreliable.