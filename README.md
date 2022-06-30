# tune-n-distill

This repository contains a pipeline to tune the mBART50 NMT pre-trained model to low-resource language pairs, and then distill the resulting system to obtain lightweight and more sustainable models.
The pipeline allows training lightweight models for the translation between English and a specific low-resource language, even if mBART50 has not been pre-trained with the low-resource language.


## Set up environment
```
git clone https://github.com/transducens/tune-n-distill.git
cd tune-n-distill
```

### Prepare directories

The directories used for the pipeline are:
* Corpora: For each language pair, it will contain the parallel corpora, the monolingual corpora and the dev/test sets.
* Models: The mBART50 initial models, mbart50-1n and mbart50-n1
* Experiments: Path where the trained systems will be stored.
```
mkdir corpora models experiments
```
For each language pair to train, create the directory with the respective corpora.

For example:
```
 corpora/en-mk/train.en
              /train.mk
              /monolingual.en
              /monolingual.mk
              /dev.en
              /dev.mk
              /test.en
              /test.mk
```

### Download mBART50-1n and mBART50-n1
```
./download-mbart50.sh models
```

### Build and run Docker container

Build image with tag name.
```
./docker-build.sh tune-n-distill:0.1
```

The container uses Volumes for persistence of the generated data and avoids increasing the size of the container. The docker-run script needs the name of the image created before and the paths of the models, corpora and experiments directories.
By default, the GPU 0 as shown by the nvidia-smi command will be used to train the systems. If you want to use another GPU, specify the GPU number with the --gpu parameter.
```
./docker-run.sh –image-tag=tune-n-distill:01 –docker-name=tune-n-distill-en-mk –models-dir=$PWD/models –experiments-dir=$PWD/experiments –corpora-dir=$PWD/corpora –gpu=1 
```

After executing the script, the container is ready to start the training.

```user@containerID:/app$```

## Train systems
The file "launch_1-n.sh" contains all the variables needed to start the training.

Lang_1 and lang_2 are the language pair for training. At the end, the result will be one model for ```lang_1-lang_2``` direction and another model for ```lang_2-lang_1``` direction. Since mBART50 uses special lang codes, it is necessary to pass both the normal code and the mBART50 code.
Lang_1 must always be ```"en en_XX"```.

The rest of the parameters have the following meaning:

* experiment-dir: Path where the trained systems will be stored.
* mBART50-dir: Path of the mBART50 models to be tuned.
* corpora-for-bicleaner-train: String with parallel corpora, separated by spaces and without language code. [More information about Bicleaner-AI training](https://github.com/bitextor/bicleaner-ai)
* corpora-for-bicleaner-dev: A development set made of very clean parallel sentences.
* parallel-corpora: String with parallel corpora, separated by spaces and without language code.
* monolingual-corpora-lang-1: String with monolingual corpora for lang-1, separated by spaces and without language code.
* monolingual-corpora-lang-2: String with monolingual corpora for lang-2, separated by spaces and without language code.
* dev-corpus: A development set made of very clean parallel sentences.
* test-corpus: A test set made of very clean parallel sentences.

All corpora must be in ```corpora/$lang_1-$lang_2```.

Example of launch_1-n.sh file and the required directories and files:

Launch_1-n.sh
```
bash launch-train.sh \
    --lang-1="en en_XX" \
    --lang-2="mk mk_MK" \
    --experiment-dir="en-mk" \
    --mBART50-dir="/models" \
    --corpora-for-bicleaner-dict="GlobalVoices Tatoeba" \
    --corpora-for-bicleaner-model="flores101_devset" \
    --parallel-corpora="GlobalVoices TED2020 Tatoeba" \
    --monolingual-corpora-lang-1="NewsCrawl" \
    --monolingual-corpora-lang-2="NewsCrawl GoURMET" \
    --dev-corpus="flores101_devset" \
    --test-corpus="flores101_testset"
```

Container directories:
```
/corpora/en-mk/GlobalVoices.en
               GlobalVoices.mk
               TED2020.en
               TED2020.mk
               Tatoeba.en
               Tatoeba.mk
               NewsCrawl.en
               NewsCrawl.mk
               GoURMET.mk
               flores101_devset.en
               flores101_devset.mk
               flores101_testset.en
               flores101_testset.mk

/models/mbart50-1n
        mbart50-n1
```

### Launch train
```
./launch_1n.sh 2> error.log
```

The different steps of the process will be stored in the directory indicated for the experiment (```--experiment-dir```).
You can find the resulting BLEU and the best mBART50 model in the file  ```/experiments/$experiment-dir/fine_tuning.result```.
In addition, each student has in its directory a file with its BLEU, chrF++ and spBLEU.
