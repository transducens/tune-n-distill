#!/bin/bash
set -euo pipefail

DIR=/experiments
CORPORA=/corpora
MODELS=/models
APP=/app/tune-n-distill

LANG_1=
LANG_2=

EXPERIMENT_DIR=
MBART50_DIR=

BICLEANER_TRAIN_CORPUS=
BICLEANER_DEV_CORPUS=

TRAIN_CORPUS=

MONO_LANG_1=
MONO_LANG_2=

DEV_CORPUS=
TEST_CORPUS=

while [ $# -gt 0 ]; do
    case "$1" in
        --lang-1=*)
            LANG_1="${1#*=}"
            ;;
        --lang-2=*)
            LANG_2="${1#*=}"
            ;;
         --experiment-dir=*)
            EXPERIMENT_DIR="${1#*=}"
            ;;
         --mBART50-dir=*)
            MBART50_DIR="${1#*=}"
            ;;
        --corpora-for-bicleaner-train=*)
            BICLEANER_TRAIN_CORPUS="${1#*=}"
            ;;
        --corpora-for-bicleaner-dev=*)
            BICLEANER_DEV_CORPUS="${1#*=}"
            ;;
        --parallel-corpora=*)
            TRAIN_CORPUS="${1#*=}"
            ;;
        --monolingual-corpora-lang-1=*)
            MONO_LANG_1="${1#*=}"
            ;;
        --monolingual-corpora-lang-2=*)
            MONO_LANG_2="${1#*=}"
            ;;
         --dev-corpus=*)
            DEV_CORPUS="${1#*=}"
            ;;
         --test-corpus=*)
            TEST_CORPUS="${1#*=}"
            ;;
        *)
            echo "Error: Invalid argument: $1"
            exit 1
  esac
  shift
done

if [ -z "$LANG_1" ] || [ -z "$LANG_2" ]  || [ -z "$EXPERIMENT_DIR" ] || [ -z "$MBART50_DIR" ] || [ -z "$BICLEANER_TRAIN_CORPUS" ] || [ -z "$BICLEANER_DEV_CORPUS" ] || [ -z "$TRAIN_CORPUS" ] || [ -z "$MONO_LANG_1" ] || [ -z "$MONO_LANG_2" ] || [ -z "$DEV_CORPUS" ] || [ -z "$TEST_CORPUS" ]; then
   printf '%s\n'  'Arguments:' \
                  '--lang-1="en en_XX"'  \
                  '--lang-2=...' \
                  '--experiment-dir=...' \
                  '--mBART50-dir=Directory which contains the directories mBart5_1n and mBart50_n1' \
                  '--corpora-for-bicleaner-train="corpus_1 corpus_2 ..."' \
                  '--corpora-for-bicleaner-dev="corpus_3 corpus_4 ..."' \
                  '--parallel-corpora="corpus_1 corpus_2 ..."' \
                  '--monolingual-corpora-lang-1=...' \
                  '--monolingual-corpora-lang-2=...' \
                  '--dev-corpus=...' \
                  '--test-corpus=...'
    exit 1  
fi

ln -s /home/user/sacrebleu $PWD/tools/sacrebleu

langs_1=($LANG_1)
LANG_1=${langs_1[0]} #normal lang code
LANG_1_MBART50=${langs_1[1]} #mBART50 lang code

langs_2=($LANG_2)
LANG_2=${langs_2[0]} #normal lang code
LANG_2_MBART50=${langs_2[1]} #mBART50 lang code

CORPORA=/corpora/$LANG_1-$LANG_2

echo "########## Training Bicleaner model ##########"
mkdir -p $DIR/$EXPERIMENT_DIR
./train-bicleaner.sh $LANG_1 $LANG_2 $DIR/$EXPERIMENT_DIR $CORPORA $BICLEANER_TRAIN_CORPUS $BICLEANER_DEV_CORPUS $MONO_LANG_2

echo "########## Prepare mBART50 corpora ##########"
if [[ ! -f $DIR/$EXPERIMENT_DIR/corpora/mono.$LANG_1_MBART50 ]]; then
   mkdir -p $DIR/$EXPERIMENT_DIR/corpora
   for mono in $MONO_LANG_1; do
      cat $CORPORA/$mono.$LANG_1 >> $DIR/$EXPERIMENT_DIR/corpora/mono.$LANG_1_MBART50
   done
   for mono in $MONO_LANG_2; do
      cat $CORPORA/$mono.$LANG_2 >> $DIR/$EXPERIMENT_DIR/corpora/mono.$LANG_2_MBART50
   done
   for corpus in $TRAIN_CORPUS; do
      cat $CORPORA/$corpus.$LANG_1 >> $DIR/$EXPERIMENT_DIR/corpora/train.$LANG_1_MBART50
      cat $CORPORA/$corpus.$LANG_2 >> $DIR/$EXPERIMENT_DIR/corpora/train.$LANG_2_MBART50
   done
   cp $CORPORA/$DEV_CORPUS.$LANG_1 $DIR/$EXPERIMENT_DIR/corpora/valid.$LANG_1_MBART50
   cp $CORPORA/$DEV_CORPUS.$LANG_2 $DIR/$EXPERIMENT_DIR/corpora/valid.$LANG_2_MBART50

   cp $CORPORA/$TEST_CORPUS.$LANG_1 $DIR/$EXPERIMENT_DIR/corpora/test.$LANG_1_MBART50
   cp $CORPORA/$TEST_CORPUS.$LANG_2 $DIR/$EXPERIMENT_DIR/corpora/test.$LANG_2_MBART50
fi
echo "########## Finetuning mBART50 with iterative-backtranslation ##########"
./iterative-backtranslation-withfinetune.sh $LANG_1_MBART50 $LANG_2_MBART50 $DIR/$EXPERIMENT_DIR $MBART50_DIR

prepare_students_dir () {
   it=$1
   transformers="back forward all"
   for transformer in $transformers; do #Prepare corpus for back, forward and all training
      echo "########## Prepare $transformer directories ##########"
      mkdir -p $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus
      mkdir -p $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus
      for corpus in $TRAIN_CORPUS; do
         cat $CORPORA/$corpus.$LANG_1 >> $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus/train.$LANG_1
         cat $CORPORA/$corpus.$LANG_2 >> $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus/train.$LANG_2

         cat $CORPORA/$corpus.$LANG_1 >> $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus/train.$LANG_1
         cat $CORPORA/$corpus.$LANG_2 >> $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus/train.$LANG_2
      done

      cp $CORPORA/$DEV_CORPUS.$LANG_1 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus/dev.$LANG_1
      cp $CORPORA/$DEV_CORPUS.$LANG_2 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus/dev.$LANG_2

      cp $CORPORA/$DEV_CORPUS.$LANG_1 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus/dev.$LANG_1
      cp $CORPORA/$DEV_CORPUS.$LANG_2 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus/dev.$LANG_2

      cp $CORPORA/$TEST_CORPUS.$LANG_1 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus/test.$LANG_1
      cp $CORPORA/$TEST_CORPUS.$LANG_2 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_1-$LANG_2/corpus/test.$LANG_2

      cp $CORPORA/$TEST_CORPUS.$LANG_1 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus/test.$LANG_1
      cp $CORPORA/$TEST_CORPUS.$LANG_2 $DIR/$EXPERIMENT_DIR/$transformer-$it-$LANG_2-$LANG_1/corpus/test.$LANG_2
   done

   echo "########## en_synthetic-n_original is required by back-en-n, all-en-n, all-n-en, forward-n-en"
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_synthetic >> $DIR/$EXPERIMENT_DIR/back-$it-$LANG_1-$LANG_2/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_original >> $DIR/$EXPERIMENT_DIR/back-$it-$LANG_1-$LANG_2/corpus/train.$LANG_2

   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_synthetic >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_1-$LANG_2/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_original >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_1-$LANG_2/corpus/train.$LANG_2

   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_synthetic >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_2-$LANG_1/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_original >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_2-$LANG_1/corpus/train.$LANG_2

   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_synthetic >> $DIR/$EXPERIMENT_DIR/forward-$it-$LANG_2-$LANG_1/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_original >> $DIR/$EXPERIMENT_DIR/forward-$it-$LANG_2-$LANG_1/corpus/train.$LANG_2

   echo "########## en_original-n_synthetic is required by back-n-en, all-en-n, all-n-en, forward-en-n"
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_original >> $DIR/$EXPERIMENT_DIR/back-$it-$LANG_2-$LANG_1/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_synthetic >> $DIR/$EXPERIMENT_DIR/back-$it-$LANG_2-$LANG_1/corpus/train.$LANG_2

   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_original >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_1-$LANG_2/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_synthetic >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_1-$LANG_2/corpus/train.$LANG_2

   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_original >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_2-$LANG_1/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_synthetic >> $DIR/$EXPERIMENT_DIR/all-$it-$LANG_2-$LANG_1/corpus/train.$LANG_2

   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_1_original >> $DIR/$EXPERIMENT_DIR/forward-$it-$LANG_1-$LANG_2/corpus/train.$LANG_1
   cat $DIR/$EXPERIMENT_DIR/synthetic_corpora/mono_n_synthetic >> $DIR/$EXPERIMENT_DIR/forward-$it-$LANG_1-$LANG_2/corpus/train.$LANG_2
}

run_students () {
   it_r=$1
   transformers="back forward all"
   for transformer in $transformers; do
      #sbatch --gres=gpu:1 train-student.sh
      ./train-student.sh $LANG_1 $LANG_2 $DIR/$EXPERIMENT_DIR/$transformer-$it_r-$LANG_1-$LANG_2
      ./train-student.sh $LANG_2 $LANG_1 $DIR/$EXPERIMENT_DIR/$transformer-$it_r-$LANG_2-$LANG_1
   done
}

#Create all, back and forward corpus
for student in 0 1 2; do
   prepare_students_dir $student
   run_students $student
done
