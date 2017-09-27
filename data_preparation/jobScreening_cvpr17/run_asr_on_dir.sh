#!/bin/bash

#Only works in croissant

ENTRADA=$1
SALIDA=$2

#cd /home/csp/repo/speech-asr/egs/luxembourgish/bin
#source path.sh
#source cmd.sh

#just in case
cd /home/jls/work/speech/speech-asr/speech-asr/egs/english/s5b/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jls/work/speech/speech-asr/speech-asr/egs/english/s5b/resources/kaldi/tools/openfst/lib/

# for fichero in $(ls $ENTRADA | grep "\.wav")
#   do
#     ./nnet6-process-wav-to-lat-and-ctm_withSAD_mono_and_stereo.sh $ENTRADA/$fichero $SALIDA
# done


for fichero in $(ls $ENTRADA | grep "\.wav")
  do
    echo $ENTRADA/$fichero 
done | parallel -j 48 ./nnet6-process-wav-to-lat-and-ctm_withSAD_mono_and_stereo.sh {} $SALIDA
