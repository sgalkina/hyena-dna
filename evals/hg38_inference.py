import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 
import pyfastx
import pandas as pd

sys.path.append(os.environ.get("SAFARI_PATH", "."))

from src.models.sequence.long_conv_lm import ConvLMHeadModel

# from transformers import AutoTokenizer, GPT2LMHeadModel
# from spacy.lang.en.stop_words import STOP_WORDS
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import numpy as np
try:
    from tokenizers import Tokenizer  
except:
    pass

# https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
# def preprocess(text):
#     text = text.replace("“", '"')
#     text = text.replace("”", '"')
#     return '\n'+text.strip()


class HG38Encoder:
    "Encoder inference for HG38 sequences"
    def __init__(self, model_cfg, ckpt_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.model, self.tokenizer = self.load_model(model_cfg, ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.model = self.model.to(self.device)

    def encode(self, seqs):
            
        results = []

        # sample code to loop thru each sample and tokenize first (char level)
        for seq in tqdm(seqs):
            
            if isinstance(self.tokenizer, Tokenizer):
                tokenized_seq = self.tokenizer.encode(seq).ids
            else:
                tokenized_seq = self.tokenizer.encode(seq)
            
            # can accept a batch, shape [B, seq_len, hidden_dim]
            logits, __ = self.model(torch.tensor([tokenized_seq]).to(device=self.device))

            # Using head, so just have logits
            results.append(logits)

        return results
        
            
    def load_model(self, model_cfg, ckpt_path):
        config = yaml.load(open(model_cfg, 'r'), Loader=yaml.FullLoader)
        model = ConvLMHeadModel(**config['model_config'])
        
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # state_dict = torch.load(ckpt_path)

        # loads model from ddp by removing prexix to single if necessary
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )

        model_state_dict = state_dict["state_dict"]

        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)

        model.load_state_dict(state_dict["state_dict"])

        # setup tokenizer
        if config['tokenizer_name'] == 'char':
            print("**Using Char-level tokenizer**")

            # add to vocab
            tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_seq_len + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
            print(tokenizer._vocab_str_to_int)
        else:
            raise NotImplementedError("You need to provide a custom tokenizer!")

        return model, tokenizer


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""
    if top_p <= 0.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
     # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
            ]
        else:
            logits_top = logits / temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_cfg",
        default=f"/home/nmb127/code/hyena-dna/configs/evals/hg38.yaml",
    )
    
    parser.add_argument(
        "--ckpt_path",
        default=f"",
        help="Path to model state dict checkpoint"
    )
        
    args = parser.parse_args()
    max_seq_len = 16384
        
    task = HG38Encoder(args.model_cfg, args.ckpt_path, max_seq_len=max_seq_len)

    # # sample sequence, can pass a list of seqs (themselves a list of chars)
    # FILENAME = 'gtdb_genomes_reps_r207/GCA/910/822/395/GCA_910822395.1_genomic.fna.gz'
    # SEQ_NAME = 'OU365335.1'
    # # FILENAME = 'contigs_2kbp.fna.gz'
    # SEQ_NAME = None
    # seqs_all = {name: seq for name, seq in pyfastx.Fastx(os.path.join('/home/nmb127/code/data/', FILENAME))}
    # for n, s in seqs_all.items():
    #     if len(s) > 20000:
    #         print(n)
    #         SEQ_NAME = n
    #         break
    # seqs = [seqs_all[SEQ_NAME]]
    # print(len(seqs[0]))

    # # sequence = seqs[0]
    # START = 5000
    # END = START+max_seq_len
    # LEN_RES = 1000
    # ground_truth = seqs[0][END-LEN_RES:END]
    # sequence = seqs[0][START:END-LEN_RES]
    # result = ''
    # for i in range(LEN_RES):
    #     sequence = seqs[0][START:END-LEN_RES+i]
    #     logits = task.encode([sequence])
    #     # print('len', len(seqs[0]))
    #     # print(logits[0].logits.shape)
    #     logits = logits[0].logits

    #     batch_preds = logits[0].argmax(axis=-1)
    #     string = task.tokenizer._vocab_int_to_str[batch_preds[-2].item()]
    #     # batch_preds = sample(logits[0], top_k=3)
    #     # string = task.tokenizer._vocab_int_to_str[batch_preds[-2].item()]
    #     # string = ''.join([task.tokenizer._vocab_int_to_str[i.item()] for i in batch_preds[0][:-1]])
    #     result += string
        # sequence += string

    # result = ''.join([np.random.choice(['A', 'G', 'T', 'C']) for _ in range(len(ground_truth))])

    df_path = pd.read_csv('/home/nmb127/code/data/genome_paths_map.csv')
    PATHS_MAP = {o: f for o, f in zip(df_path['accession'], df_path['0'])}
    N = 1000
    KEYS = list(PATHS_MAP.keys())[:N]
    START = 0
    max_seq_len = 16384
    END = START+max_seq_len
    c = 0
    for key in KEYS:
        for name, s in pyfastx.Fastx(os.path.join('/home/nmb127/code/data/', PATHS_MAP[key])):
            if len(s) > max_seq_len:
                seq = s
                break
        sequence = seq[START:END]
        logits = task.encode([sequence])
        logits = logits[0].logits
        batch_preds = logits[0].argmax(axis=-1)
        string = task.tokenizer._vocab_int_to_str[batch_preds[-2].item()]
        c += int(string == seq[END])

    print('Acc', c / N)
    # print(batch_preds.shape)

    # breakpoint()

    