#!/bin/bash
set -euo pipefail

#./train-bicleaner.sh lang_1 lang_2 corpora_dir $DICT_CORPUS $MODEL_CORPUS

lang1=$1
lang2=$2

EXPERIMENT_DIR=$3
CORPORA=$4

TRAIN_CORPUS=$5
DEV_CORPUS=$6

MONOLINGUAL_CORPUS=$7

source /opt/conda/etc/profile.d/conda.sh
conda activate bicleaner-ai

cd $EXPERIMENT_DIR
mkdir -p bicleaner-$lang1-$lang2
cd bicleaner-$lang1-$lang2
mkdir -p models/$lang1-$lang2

### TRAIN BICLEANER-AI 
for corpus in $TRAIN_CORPUS; do
	cat $CORPORA/$corpus.$lang1 >> train.$lang1
	cat $CORPORA/$corpus.$lang2 >> train.$lang2
done

for corpus in $DEV_CORPUS; do
    cat $CORPORA/$corpus.$lang1 >> dev.$lang1
    cat $CORPORA/$corpus.$lang2 >> dev.$lang2
done
paste train.$lang1 train.$lang2 > train.$lang1-$lang2
rm train.$lang1
rm train.$lang2

#Clean bicleaner-ai data
awk -F "\t" '{if ($1 !="" && $2 != "") print $0}' train.$lang1-$lang2 > train_without_empty_lines.$lang1-$lang2
rm train.$lang1-$lang2
mv train_without_empty_lines.$lang1-$lang2 train.$lang1-$lang2

lines=$(awk 'END{print NR}' train.$lang1-$lang2)
if awk "BEGIN {exit !($lines > 600000)}"; then
    shuf train.$lang1-$lang2 > train.shuffled.$lang1-$lang2
    head -600000 train.$lang1-$lang2 > small_train.$lang1-$lang2
    rm train.shuffled.$lang1-$lang2
    rm train.$lang1-$lang2
    mv small_train.$lang1-$lang2 train.$lang1-$lang2
fi

paste dev.$lang1 dev.$lang2 > dev.$lang1-$lang2
rm dev.$lang1
rm dev.$lang2

cat $CORPORA/$MONOLINGUAL_CORPUS.$lang2 \
    | sacremoses -l $lang2 tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \
    | grep -v "[[:space:]]*1" \
    | gzip >  models/$lang1-$lang2/wordfreq-$lang2.gz

bicleaner-ai-train \
    --classifier_type xlmr \
    --batch_size 64 \
    --steps_per_epoch 1000 \
    --epochs 15 --patience 4 \
    -m models/$lang1-$lang2 -s $lang1 -t $lang2 \
    -F models/$lang1-$lang2/wordfreq-$lang2.gz \
    --parallel_train train.$lang1-$lang2 \
    --parallel_dev dev.$lang1-$lang2 #\
    #--lm_file_sl model.$lang1-$lang2.$lang1 --lm_file_tl model.$lang1-$lang2.$lang2

conda deactivate

cd /app/
