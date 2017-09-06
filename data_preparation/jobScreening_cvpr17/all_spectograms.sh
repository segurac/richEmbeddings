#!/bin/bash


INDIR=$1
OUTDIR=$2


for fichero in $(ls $INDIR | grep wav)
    do
    
        echo $fichero
        ./extract_spectograms.py $INDIR/$fichero $OUTDIR/${fichero}.fbank.pickle
done

