#!/bin/bash

for fichero in $(ls transcripts)
  do
    cat transcripts/$fichero |  python3 -c 'import sys; print( sys.stdin.read().lower() )' | sed 's@-@\ @g' | sed 's@–@\ @g' |  sed -e 's@\.\.\.@@g' -e 's@\[[^][]*\]@\ [vocalized-noise]\ @g' -e 's@\.\.@.@g' -e 's@ \. @. @g' -e 's@\.@\ @g' | sed -e 's@_ @\ @g' -e 's@_$@@g' | sed -e 's@[,!\"?]@\ @g'  -e 's@\ \ @\ @g' > transcripts/${fichero}.clean
    
done
