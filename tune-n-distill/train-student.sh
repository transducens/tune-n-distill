#!/bin/bash 

set -euo pipefail

if [ $# -lt 3 ]
then
  echo "Wrong number of arguments"
  exit 1
fi

lang1=$1
lang2=$2
permanentDir=$3

maxLegthAfterBpe=100

source train-steps-fairseq.sh
#########################################

train_sentencepiece

apply_sentencepiece train $lang1 $permanentDir/spm/spm.$lang1.$lang2.model
apply_sentencepiece train $lang2 $permanentDir/spm/spm.$lang1.$lang2.model

apply_sentencepiece dev $lang1 $permanentDir/spm/spm.$lang1.$lang2.model
apply_sentencepiece dev $lang2 $permanentDir/spm/spm.$lang1.$lang2.model

apply_sentencepiece test $lang1 $permanentDir/spm/spm.$lang1.$lang2.model
apply_sentencepiece test $lang2 $permanentDir/spm/spm.$lang1.$lang2.model

preprocess train
train

translate_test_spm train

report_spm train
