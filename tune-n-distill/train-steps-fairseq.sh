#!/bin/bash 

set -euo pipefail

gpuId=0
temp=/tmp

trainArgs="--arch transformer --share-all-embeddings  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --weight-decay 0  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-7 --lr 0.0007 --min-lr 1e-9  --save-interval-updates 5000  --patience 6 --no-progress-bar --max-tokens 4000 --eval-bleu --eval-tokenized-bleu --eval-bleu-args '{\"beam\":5,\"max_len_a\":1.2,\"max_len_b\":10}' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-best-checkpoints 1 --keep-interval-updates 1 --no-epoch-checkpoints"

TOOLS=$PWD/tools

train_sentencepiece () {
  if [[ ! -f $permanentDir/spm/spm.$lang1.$lang2.model ]]; then
    mkdir -p $permanentDir/spm
    cd $permanentDir/spm
    cat $permanentDir/corpus/train.$lang1 $permanentDir/corpus/train.$lang2 | shuf > train_spm
    python $TOOLS/train_spm.py --input $permanentDir/spm/train_spm --model_prefix spm.$lang1.$lang2
    rm train_spm
    cd /app
  fi
}

apply_sentencepiece () {
  prefix=$1
  lang=$2

  sp_model=$3
  echo "sentence_piece $prefix $lang ######################"

  if [ ! -e $permanentDir/corpus/$prefix.$lang ]
  then
    echo "sentence_piece: ERROR: File $permanentDir/corpus/$prefix.$lang does not exist"
    exit 1
  fi

  if [[ ! -f $permanentDir/corpus/$prefix.spm.$lang ]]; then
    python $TOOLS/spm_encode.py --model $sp_model --output_format=piece --inputs=$permanentDir/corpus/$prefix.$lang --outputs=$permanentDir/corpus/$prefix.spm.$lang
  fi
}

preprocess () {
  echo "make_data_for_training $@ ######################"
  echo "preprocess -s $lang1 -t $lang2"

  for tag in "$@"
  do
    if [ ! -e $permanentDir/corpus/$tag.spm.$lang1 ]
    then
      echo "make_data_for_training: ERROR: File $permanentDir/corpus/$tag.spm.$lang1 does not exist"
      exit 1
    fi

    if [ ! -e $permanentDir/corpus/$tag.spm.$lang2 ]
    then
      echo "make_data_for_training: ERROR: File $permanentDir/corpus/$tag.spm.$lang2 does not exist"
      exit 1
    fi
  done

  fairseq-preprocess -s $lang1 -t $lang2  --trainpref $permanentDir/corpus/train.spm \
                     --validpref $permanentDir/corpus/dev.spm \
                     --testpref $permanentDir/corpus/test.spm \
                     --destdir $permanentDir/model/data-bin-train --workers 16 --joined-dictionary
}

translate_test_spm () {
  tag=$1
  echo "translate_test $tag $lang1 - $lang2 ######################"

  if [ ! -e $permanentDir/model/checkpoints/$tag.checkpoint_best.pt ]
  then
    echo "translate_test_fairseq: ERROR: File $permanentDir/model/checkpoints/$tag.checkpoint_best.pt does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/test.spm.$lang1 ]
  then
    echo "translate_test_fairseq: ERROR: File $permanentDir/corpus/test.spm.$lang1 does not exist"
    exit 1
  fi

  if [ ! -d $permanentDir/model/data-bin-$tag ]
  then
    echo "train_nmt_fairseq: ERROR: Folder $permanentDir/model/data-bin-$tag does not exist"
    exit 1
  fi

  mkdir -p $permanentDir/eval/

  CUDA_VISIBLE_DEVICES=0 fairseq-interactive  --input $permanentDir/corpus/test.spm.$lang1 --path $permanentDir/model/checkpoints/$tag.checkpoint_best.pt \
                                              $permanentDir/model/data-bin-$tag --remove-bpe 'sentencepiece' | grep '^H-' | cut -f 3 > $permanentDir/eval/test.output-$tag
}

report_spm () {
  tag=$1
  echo "report $tag ######################"

  if [ ! -e $permanentDir/eval/test.output-$tag ]
  then
    echo "report: ERROR: File $permanentDir/eval/test.output-$tag does not exist"
    exit 1
  fi

  if [ ! -e $permanentDir/corpus/test.$lang2 ]
  then
    echo "report: ERROR: File $permanentDir/corpus/test.$lang2 does not exist"
    exit 1
  fi
  python $TOOLS/sacrebleu/sacrebleu/sacrebleu.py $permanentDir/corpus/test.$lang2 < $permanentDir/eval/test.output-$tag --metrics bleu chrf | cut -f 3 -d  ' ' > $permanentDir/eval/report-$tag
	python $TOOLS/sacrebleu/sacrebleu/sacrebleu.py $permanentDir/corpus/test.$lang2 < $permanentDir/eval/test.output-$tag --tokenize spm | cut -f 3 -d  ' ' >> $permanentDir/eval/report-$tag
}

train () {
  echo "train######################"
  
  if [ ! -d $permanentDir/model/data-bin-train ]
  then
    echo "train_fairseq: ERROR: Folder $permanentDir/model/data-bin-train does not exist"
    exit 1
  fi

  echo "Training args: $trainArgs"
  echo "See $permanentDir/model/train.log for details"  
  
  eval "CUDA_VISIBLE_DEVICES=0 fairseq-train $trainArgs --seed $RANDOM --save-dir $permanentDir/model/checkpoints $permanentDir/model/data-bin-train &> $permanentDir/model/train.log"

  mv $permanentDir/model/checkpoints/checkpoint_best.pt $permanentDir/model/checkpoints/train.checkpoint_best.pt
  rm -fr $permanentDir/model/checkpoints/checkpoint* 
}

