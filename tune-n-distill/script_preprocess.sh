#!/bin/bash

SOURCE=
TARGET=
MODEL_DIR=
EXPERIMENT_DIR=

MONO=0

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
	--mono=*)
	    MONO="${1#*=}"
	    ;;
        *)
            echo "Error: Invalid argument: $1"
            exit 1
  esac
  shift
done

if [ -z "$SOURCE" ] || [ -z "$TARGET" ]  || [ -z "$MODEL_DIR" ] || [ -z "$EXPERIMENT_DIR" ]; then
    echo "Arguments: --source=... --target=... --model-dir=... --experiment-dir=... [--mono={0,1}]"
    exit 1  
fi

# Binarize data:
if [ "$MONO" == "0" ]; then
	fairseq-preprocess --source-lang $SOURCE --target-lang $TARGET --trainpref $EXPERIMENT_DIR/corpora/train.spm --validpref $EXPERIMENT_DIR/../corpora/valid.spm --testpref $EXPERIMENT_DIR/../corpora/test.spm --thresholdsrc 0 --thresholdtgt 0 --destdir $EXPERIMENT_DIR/data_bin --srcdict $MODEL_DIR/dict.txt --tgtdict $MODEL_DIR/dict.txt >> $SOURCE-$TARGET.log
else
	fairseq-preprocess --source-lang $SOURCE --target-lang $TARGET --validpref $EXPERIMENT_DIR/spm/mono.spm --thresholdsrc 0 --thresholdtgt 0 --destdir $EXPERIMENT_DIR/data_bin_mono --srcdict $MODEL_DIR/dict.txt --tgtdict $MODEL_DIR/dict.txt >> $SOURCE-$TARGET.log
fi
