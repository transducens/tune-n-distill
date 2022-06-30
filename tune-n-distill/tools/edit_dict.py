import argparse
from glob import glob
import os
from typing import List
import torch
from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line

# adapted from https://github.com/pytorch/fairseq/issues/2120#issuecomment-647429120
# and https://github.com/pytorch/fairseq/issues/2120#issuecomment-647429120

# padding_factor can be used to pad the dictionary size to be a multiple of 8, 
# which is important on some hardware (e.g., Nvidia Tensor Cores)

def pad_dict(d: Dictionary, num_extra_symbols: int, padding_factor: int = 8) -> None:
    i = 0
    while (len(d) + num_extra_symbols) % padding_factor != 0:
        symbol = f"madeupword{i:04d}"
        d.add_symbol(symbol, n=0)
        i += 1

def load_dict(path: str) -> Dictionary:
    d = Dictionary.load(path)
    return d

# Make dictionary from corpus. The corpus must be tokenised with SentencePiece.
# Using the SentencePiece model of mBART50, the result will be a smaller vocab, keeping the learned for the corresponding embeddings.
# Using a new SentencePiece model, all new tokens will be initialised like the <unk> embedding.
def edit_dict(args) -> None:
    langs = args.langs.split(",")
    custom_dict = Dictionary()
    for data_path in glob(args.corpus_data):
        Dictionary.add_file_to_dictionary(data_path, custom_dict, tokenize_line, 4)
    for i in langs: # add language codes:
        symbol = f"__{i}__"
        custom_dict.add_symbol(symbol, n=1)
    pad_dict(custom_dict, len(langs)+1)
    # 0 changed to 8, just to make sure madeupwords are correctly added:
    custom_dict.finalize(padding_factor=8)
    custom_dict.save(args.ft_dict)

# Edit embeddings 
def edit_model(args) -> None:
    #langs = args.langs.split(",")
    #pre_dict = load_dict(langs, os.path.join(args.pre_train_dir, "dict.txt"))
    #ft_dict = load_dict(langs, args.ft_dict)
    pre_dict = load_dict(os.path.join(args.pre_train_dir, "dict.txt"))
    ft_dict = load_dict(args.ft_dict)

    # GPU or CPU:
    #data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
    data = torch.load(os.path.join(args.pre_train_dir, "model.pt"),map_location=torch.device('cpu'))
    model = data["model"]

    mapping: List[int] = []
    new_words : List[int] = []

    print("Original vocabulary size: ",len(pre_dict))
    print("Filtered vocabulary size: ",len(ft_dict))

    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))

    keys=["encoder.embed_tokens.weight","decoder.embed_tokens.weight"] # For fine tuned mBART50: ,"decoder.output_projection.weight"]

    for name in keys:
        pre_tensor: torch.Tensor = model[name]
        print ("Size of original",name,":",pre_tensor.size())
        embed_dim= pre_tensor.size()[1]
        ft_tensor = torch.zeros(
            [len(ft_dict), embed_dim], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            # print(f"Keeping embedding {pre_i}.")
            # for unknown words the embedding of <unk> is used
            ft_tensor[ft_i] = pre_tensor[pre_i]
        model[name] = ft_tensor
        post_tensor: torch.Tensor = model[name]
        assert post_tensor.size()[1]==embed_dim
        print ("Size of filtered ",name,": ",post_tensor.size())

    torch.save(data, args.output)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabulary from corpus data and trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--corpus-data", type=str, required=True, help="The path pattern (glob) to all *tokenized* corpus files (train, test, val).")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.")
    parser.add_argument("--output", type=str, required=True, help="The trimmed model.")
    args = parser.parse_args()

    edit_dict(args)
    edit_model(args)

if __name__ == "__main__":
    main()
