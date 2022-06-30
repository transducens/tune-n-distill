#!/bin/bash
set -euo pipefail

#./classify-corpus.sh lang1 lang2 corpus

lang1=$1
lang2=$2
dir=$3
corpus_1=$4
corpus_2=$5
corpus=$6

cd $dir/bicleaner-$lang1-$lang2
paste $corpus_1 $corpus_2 > $corpus.$lang1-$lang2

source /opt/conda/etc/profile.d/conda.sh
conda activate bicleaner-ai

#Classify with bicleaner-ai
bicleaner-ai-classify --scol 1 --tcol 2 $corpus.$lang1-$lang2  $corpus.$lang1-$lang2.classified models/$lang1-$lang2/metadata.yaml

conda deactivate
PATH=/usr/local/nvm/versions/node/v15.2.1/bin:/home/user/.local/bin:/opt/conda/bin:/opt/cmake-3.14.6-Linux-x86_64/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin

cd /app
