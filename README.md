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

## Citing this work

If you use this repository as part of your developments, please cite it as follows:

```
@inproceedings{galiano-jimenez-etal-2023-exploiting,
    title = "Exploiting large pre-trained models for low-resource neural machine translation",
    author = "Galiano-Jim{\'e}nez, Aar{\'o}n  and
      S{\'a}nchez-Mart{\'i}nez, Felipe  and
      S{\'a}nchez-Cartagena, V{\'i}ctor M.  and
      P{\'e}rez-Ortiz, Juan Antonio",
    editor = "Nurminen, Mary  and
      Brenner, Judith  and
      Koponen, Maarit  and
      Latomaa, Sirkku  and
      Mikhailov, Mikhail  and
      Schierl, Frederike  and
      Ranasinghe, Tharindu  and
      Vanmassenhove, Eva  and
      Vidal, Sergi Alvarez  and
      Aranberri, Nora  and
      Nunziatini, Mara  and
      Escart{\'i}n, Carla Parra  and
      Forcada, Mikel  and
      Popovic, Maja  and
      Scarton, Carolina  and
      Moniz, Helena",
    booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.7/",
    pages = "59--68",
    abstract = "Pre-trained models have drastically changed the field of natural language processing by providing a way to leverage large-scale language representations to various tasks. Some pre-trained models offer general-purpose representations, while others are specialized in particular tasks, like neural machine translation (NMT). Multilingual NMT-targeted systems are often fine-tuned for specific language pairs, but there is a lack of evidence-based best-practice recommendations to guide this process. Moreover, the trend towards even larger pre-trained models has made it challenging to deploy them in the computationally restrictive environments typically found in developing regions where low-resource languages are usually spoken. We propose a pipeline to tune the mBART50 pre-trained model to 8 diverse low-resource language pairs, and then distil the resulting system to obtain lightweight and more sustainable models. Our pipeline conveniently exploits back-translation, synthetic corpus filtering, and knowledge distillation to deliver efficient, yet powerful bilingual translation models 13 times smaller than the original pre-trained ones, but with close performance in terms of BLEU."
}
```

A `CITATION.cff` file is also included in this repository.
