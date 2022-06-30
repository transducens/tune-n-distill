import sentencepiece as spm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="input files to train"
    )
    parser.add_argument(
        "--model_prefix", required=True, help="sentencepiece model name"
    )
    args = parser.parse_args()
    arguments = '--input=' + args.input +' --model_prefix=' + args.model_prefix + ' --vocab_size=10000 --model_type=bpe'
    spm.SentencePieceTrainer.train(arguments)

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.




if __name__ == "__main__":
    main()