#!/bin/bash

ENTRADA=$1
SALIDA=$2

#just in case
cd /home/jls/work/speech/speech-asr/speech-asr/egs/english/s5b/bin

for fichero in $(ls $ENTRADA | grep "\.wav")
  do
    ./nnet6-process-wav-to-lat-and-ctm_withSAD_mono_and_stereo.sh $ENTRADA/$fichero $SALIDA
done
