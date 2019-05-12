# speech-tagging-tool

## About

Tool for labelling phonemes and words with start and stop timestamps based on
neural network's speech-to-phoneme engine.

## Quick links

* [Architecture](#architecture)
* [Dataset](#dataset)
* [Future development](#future-development)

## Architecture

System is based on recurrent neural networks with connectionist temporal
classification (CTC) loss function. As an input networks gets 26 features (MFCC +
deltas) obtained from audio file.

## Dataset

We are using TIMIT dataset, which you can download from here:
https://goo.gl/l0sPwz

## Future development

In future we want to train model on Polish dataset.
