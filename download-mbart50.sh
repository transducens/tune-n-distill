#!/bin/bash

if [ $# -eq 0 ] # $1 = dir
  then
    echo "No arguments supplied. Indicate the path where you want to download the models."
    exit
fi

for model in 1n n1; do
  TYPE=mbart50.ft.$model
  DIR=$1/mbart50-$model
  mkdir -p $DIR

  # Get model:
  wget -nc -P $DIR https://dl.fbaipublicfiles.com/fairseq/models/mbart50/$TYPE.tar.gz

  tar -xzvf $DIR/$TYPE.tar.gz -C $DIR
  ln -sfr $DIR/$TYPE/model.pt $DIR/model.pt
  ln -sfr $DIR/$TYPE/sentence.bpe.model $DIR/spm.model
  paste -d, -s $DIR/$TYPE/ML50_langs.txt >$DIR/langs.txt

  # Without adding this, fairseq-generate complains first "cannot find language token __en_XX__ in the dictionary" and then, when all languages added but with wrong padding words "size mismatch for decoder.output_projection.weight: copying a param with shape torch.Size([250054, 1024]) from checkpoint, the shape in current model is torch.Size([250060, 1024])". In principle, --langs or --dict-lang are for this, but I could not make it work.
  # All dictionaries are equal; use English, for example...
  cat $DIR/$TYPE/dict.en_XX.txt dict_extension_langs_mbart50.txt >$DIR/extended_dict.txt
  ln -sfr $DIR/extended_dict.txt $DIR/dict.txt

  echo "Model $TYPE stored in $DIR"
done
