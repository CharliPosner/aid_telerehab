#!/bin/bash

cd F_Subjects_data

for SUBJECT in {1..90}
do
    NAME=$(basename F_v3d_Subject_$SUBJECT)

    echo $NAME

    python3 ../seq2img.py $NAME.mat $SUBJECT 120
done
