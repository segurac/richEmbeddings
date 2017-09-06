#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import pickle
import numpy as np

from scipy.io import wavfile
import python_speech_features as fextract


audio_filename = sys.argv[1]
features_filename = sys.argv[2]


rate, sig = wavfile.read(audio_filename)


fbank_feat = fextract.logfbank(sig,samplerate=rate)


with open(features_filename, 'wb') as stream:
    pickle.dump(fbank_feat, stream)

