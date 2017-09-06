#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import pickle
import numpy as np

from scipy.io import wavfile
import python_speech_features as fextract


audio_filename_list = sys.argv[1]
features_filename_dir = sys.argv[2]


with open(audio_filename_list, 'r') as list_file:
    for line in list_file:
        audio_filename = line.strip()
        
        features_filename = features_filename_dir + audio_filename.split('/')[-1] + '.fbank.pickle'
        print("Processing", audio_filename, features_filename)

        rate, sig = wavfile.read(audio_filename)


        fbank_feat = fextract.logfbank(sig,samplerate=rate)


        with open(features_filename, 'wb') as stream:
            pickle.dump(fbank_feat, stream)

