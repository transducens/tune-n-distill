#!/bin/bash

python edit_dict.py --corpus-data /corpora/en-ky/en-ky.corpora.spm --langs $(cat /models/mbart50-n1/langs.txt) --ft-dict /models/TRIM/en-ky/mbart50-n1/dict.txt --pre-train-dir /models/mbart50-n1 --output /models/TRIM/en-ky/mbart50-n1/model.pt
python edit_dict.py --corpus-data /corpora/en-ky/en-ky.corpora.spm --langs $(cat /models/mbart50-1n/langs.txt) --ft-dict /models/TRIM/en-ky/mbart50-1n/dict.txt --pre-train-dir /models/mbart50-1n --output /models/TRIM/en-ky/mbart50-1n/model.pt
