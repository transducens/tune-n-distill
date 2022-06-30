#!/bin/bash

SOURCE=
TARGET=
MODEL_DIR=
EXPERIMENT_DIR=
MODEL_TYPE=
MAX_TOKENS=4000

while [ $# -gt 0 ]; do
    case "$1" in
        --source=*)
            SOURCE="${1#*=}"
            ;;
        --target=*)
            TARGET="${1#*=}"
            ;;
        --model-dir=*)
            MODEL_DIR="${1#*=}"
            ;;
        --experiment-dir=*)
            EXPERIMENT_DIR="${1#*=}"
            ;;
        --max-tokens=*)
            MAX_TOKENS="${1#*=}"
            ;;
        --model-type=*)
            MODEL_TYPE="${1#*=}"
            ;;
        *)
            echo "Error: Invalid argument: $1"
            exit 1
  esac
  shift
done

if [ -z "$SOURCE" ] || [ -z "$TARGET" ]  || [ -z "$MODEL_DIR" ] || [ -z "$EXPERIMENT_DIR" ] || [ -z "$MODEL_TYPE" ]; then
    echo "Arguments: --source=... --target=... --model-dir=... --experiment-dir=... --model-type=={mbart50,m2m-small,m2m-medium,m2m-large} [--max-tokens=...]"
    exit 1  
fi

MBART50_PARAMETERS="--layernorm-embedding"

if [ "$MODEL_TYPE" == "mbart50" ]; then
    PARAMETERS="--arch mbart_large $MBART50_PARAMETERS"
else
    echo "Wrong model type: $MODEL_TYPE"
    exit 1  
fi

mkdir -p $EXPERIMENT_DIR/checkpoint
cp $MODEL_DIR/langs.txt $MODEL_DIR/dict.txt $MODEL_DIR/spm.model $EXPERIMENT_DIR/checkpoint

fairseq-train $EXPERIMENT_DIR/data_bin \
  --finetune-from-model $MODEL_DIR/model.pt \
  --save-dir $EXPERIMENT_DIR/checkpoint \
  --arch mbart_large \
  --task translation_multi_simple_epoch \
  --encoder-normalize-before \
  --langs $(cat $MODEL_DIR/langs.txt) \
  --lang-pairs "$SOURCE-$TARGET" \
  --max-tokens $MAX_TOKENS \
  --decoder-normalize-before \
  --sampling-method temperature \
  --sampling-temperature 1.5 \
  --encoder-langtok "src" \
  --decoder-langtok  \
  --decoder-attention-heads 16 \
  --decoder-layerdrop 0 \
  --encoder-attention-heads 16 \
  --encoder-layerdrop 0 \
  --best-checkpoint-metric loss \
  --keep-best-checkpoints 1 \
  --keep-last-epochs 1 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.2 \
  --optimizer adam \
  --adam-eps 1e-08 \
  --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --lr 0.0006 \
  --warmup-updates 2500 \
  --max-update 40000 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --weight-decay 0.0 \
  --update-freq 2 \
  --save-interval 9999999 \
  --validate-interval 9999999 \
  --save-interval-updates 300 \
  --validate-interval-updates 300 \
  --keep-interval-updates 1 \
  --no-epoch-checkpoints \
  --seed 222 \
  --log-format simple \
  --log-interval 1 \
  --patience 10 \
  --activation-dropout 0.0 \
  --activation-fn relu \
  --adaptive-softmax-dropout 0 \
  --clip-norm 0.0 \
  --dataset-impl "mmap" \
  --ddp-backend "c10d" \
  --fp16 \
  --lang-tok-style "multilingual" \
  --required-batch-size-multiple 8 \
  --share-all-embeddings \
  --share-decoder-input-output-embed \
  --layernorm-embedding &> $EXPERIMENT_DIR/checkpoint/train.log

mv $EXPERIMENT_DIR/checkpoint/checkpoint_best.pt $EXPERIMENT_DIR/checkpoint/best.pt
mv $EXPERIMENT_DIR/checkpoint/checkpoint_last.pt $EXPERIMENT_DIR/checkpoint/last.pt
ln -sfr $EXPERIMENT_DIR/checkpoint/best.pt $EXPERIMENT_DIR/checkpoint/model.pt
rm $EXPERIMENT_DIR/checkpoint/checkpoint*