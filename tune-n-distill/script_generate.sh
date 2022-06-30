#!/bin/bash

SOURCE=
TARGET=
MODEL_DIR=
EXPERIMENT_DIR=
MODEL_TYPE=
INTERACTIVE=0
MAX_TOKENS=3000

SUBSET="test"
DATA_BIN=
OUTPUT=

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
        --interactive*)
            INTERACTIVE="${1#*=}"
            ;;
	--gen-subset*)
	    SUBSET="${1#*=}"
	    ;;
        *)
            echo "Error: Invalid argument: $1"
            exit 1
  esac
  shift
done

if [ -z "$SOURCE" ] || [ -z "$TARGET" ]  || [ -z "$MODEL_DIR" ] || [ -z "$EXPERIMENT_DIR" ] || [ -z "$MODEL_TYPE" ]; then
    echo "Arguments: --source=... --target=... --model-dir=... --experiment-dir=... --model-type=={mbart50,m2m-small,m2m-medium,m2m-large} [--max-tokens=... --interactive={1,0} --gen-subset=...]"
    exit 1  
fi

MBART50_PARAMETERS=

if [ "$MODEL_TYPE" == "mbart50" ]; then
    PARAMETERS="$MBART50_PARAMETERS"
else
    echo "Wrong model type: $MODEL_TYPE"
    exit 1  
fi

if [ "$INTERACTIVE" == "0" ]; then
    if [ "$SUBSET" == "valid" ]; then
        DATA_BIN="data_bin"
        OUTPUT="valid"
    else
        DATA_BIN="data_bin_mono"
        OUTPUT="mono"
    fi
    fairseq-generate $EXPERIMENT_DIR/$DATA_BIN --path $MODEL_DIR/model.pt --fixed-dictionary $MODEL_DIR/dict.txt --source-lang $SOURCE --target-lang $TARGET --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --decoder-langtok --encoder-langtok src --gen-subset $SUBSET --max-tokens $MAX_TOKENS --langs $(cat $MODEL_DIR/langs.txt) --lang-pairs "$SOURCE-$TARGET" --max-source-positions 2000 --max-target-positions 2000 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn > $EXPERIMENT_DIR/$OUTPUT.out.$TARGET #| tee $EXPERIMENT_DIR/$OUTPUT.out.$TARGET
else
    fairseq-interactive $EXPERIMENT_DIR/data_bin --path $MODEL_DIR/model.pt --fixed-dictionary $MODEL_DIR/dict.txt --source-lang $SOURCE --target-lang $TARGET --remove-bpe 'sentencepiece' --scoring sacrebleu --beam 5 --task translation_multi_simple_epoch --decoder-langtok --encoder-langtok src --max-tokens $MAX_TOKENS --langs $(cat $MODEL_DIR/langs.txt) --lang-pairs "$SOURCE-$TARGET" $PARAMETERS --input $EXPERIMENT_DIR/spm/mono.spm.$SOURCE --fp16 --dataset-impl mmap --distributed-world-size 1 --distributed-no-spawn > $EXPERIMENT_DIR/int.out.$TARGET #| tee $EXPERIMENT_DIR/int.out.$TARGET
fi