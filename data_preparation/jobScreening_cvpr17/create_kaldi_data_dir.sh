#!/bin/bash


INPUT_TRS_DIR=$1
INPUT_TRS_DIR=$(readlink -f $INPUT_TRS_DIR)
INPUT_AUDIO_DIR=$2
INPUT_AUDIO_DIR=$(readlink -f $INPUT_AUDIO_DIR)
OUTPUT_DIR=$3


SCRIPT_PATH=$(dirname $0)
echo $SCRIPT_PATH
SCRIPT_PATH=$(readlink -f $SCRIPT_PATH)
echo $SCRIPT_PATH


mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR
rm files_unsorted.txt
rm text
rm segments
rm reco2file_and_channel
rm utt2spk
rm wav.scp


# $SCRIPT_PATH/parse_ebos_trs.py $INPUT_TRS_DIR | while read linea
for fichero_id in $(ls ../videos)
  do
      segment1_float=0.0
      segment1=0000000
      segment2_float=$(soxi -D $INPUT_AUDIO_DIR/${fichero_id}.wav)
      segment2=$(python -c "print '%07d' % (($segment2_float)*100,)")
#       texto=$(echo $linea | cut -d' ' -f6- | grep -v -e "^o.:\ " -e "^o:\ " -e "%" | sed -e 's@^.:\ *@@g' -e 's@^..:\ *@@g' -e 's@:@@g' -e 's@^x.\ @@g' -e 's@^x\ @@g' -e 's@\ \+@\ @g')
      texto=$(cat $INPUT_TRS_DIR/${fichero_id}.txt)
      texto=$(echo $texto | python3 -c 'import sys; print( sys.stdin.read().lower() )' | sed 's@-@\ @g' | sed 's@–@\ @g')
      texto=$(echo $texto | sed -e 's@\.\.\.@@g' -e 's@\[[a-z]* ..:..:..]@[vocalized-noise] [vocalized-noise] [vocalized-noise]@g' -e 's@\[[a-z0-9]* [a-z0-9]* ..:..:..]@[vocalized-noise] [vocalized-noise] [vocalized-noise]@g' -e 's@\.\.@.@g' -e 's@ \. @. @g' -e 's@\.@\ @g')
      segment_total=${fichero_id}_${segment1}_${segment2}
      
      if [[ ! -z "${texto// }" ]]
        then
#         echo $segment1 $segment2 $segment1_float $segment2_float
          echo $fichero_id >> files_unsorted.txt
          echo $segment_total $texto >> text
          echo $segment_total $fichero_id $segment1_float $segment2_float >> segments
          echo "$fichero_id $fichero_id A" >> reco2file_and_channel
          echo $segment_total $fichero_id >> utt2spk
      fi
done

cat files_unsorted.txt | sort -u | while read nombre
    do
        echo "$nombre ${INPUT_AUDIO_DIR}/${nombre}.wav" >> wav.scp
    
done

# sed -i 's@\[...\]@<garbage>@g' text
sed -i -e 's@\[hes]@<garbage>@g' -e 's@\[spk]@<garbage>@g' -e 's@\[noi]@<noise>@g' text

sed -e 's@_ @\ @g' -e 's@_$@@g' -i text
sed -i 's@[0-9]\(\ x\ *$\)@\ <sil>@g' text
sed -i -e 's@[,!\"?]@\ @g'  -e 's@\ \ @\ @g' text


