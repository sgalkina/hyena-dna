
#@title Huggingface Pretrained Wrapper

"""

This is script is a simple HuggingFace wrapper around a HyenaDNA model, to enable a one click example
of how to load the pretrained weights and get embeddings.

It will instantiate a HyenaDNA model (model class is in the `standalone_hyenadna.py`), and handle the downloading of pretrained weights from HuggingFace.

Check out the colab notebook for a simpler and more complete walk through of how to use HyenaDNA with pretrained weights.

"""


import json
import os
import subprocess
import torch
# import transformers
from transformers import PreTrainedModel
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
import pandas as pd
import numpy as np
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional
from functools import partial
from torch import Tensor
from torchvision.ops import StochasticDepth
from collections import namedtuple
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from random import randrange
import pyfastx


# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        # pretrained_model_name_or_path = os.path.join(path, model_name)
        # if os.path.isdir(pretrained_model_name_or_path) and download == False:
        #     if config is None:
        #         config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        # else:
        #     hf_url = f'https://huggingface.co/LongSafari/{model_name}'

        #     subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
        #     command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
        #     subprocess.run(command, shell=True)

        #     if config is None:
        #         config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(
            d_model=256, 
            n_layer=int(sys.argv[3]), 
            d_inner=1024, 
            vocab_size=12, 
            layer=dict(l_max=16384+2, emb_dim=5),
            pad_vocab_size_multiple=8,
            use_head=use_head, 
            n_classes=n_classes,
            )  # the new model format
        loaded_ckpt = torch.load(
            # os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            sys.argv[1],
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        # if config.get("checkpoint_mixer", False):
        #     checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        # else:
        #     checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=False)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model




####################################################################################################




"""# Inference (450k to 1M tokens)!

If all you're interested in is getting embeddings on long DNA sequences
(inference), then we can do that right here in Colab!


*   We provide an example how to load the weights from Huggingface.
*   On the free tier, which uses a
T4 GPU w/16GB of memory, we can process 450k tokens / nucleotides.
*   For processing 1M tokens, you'll need an A100, which Colab offers as a paid tier.
*   (Don't forget to run the entire notebook above too)

--

To pretrain or fine-tune the 1M long sequence model (8 layers, d_model=256),
you'll need 8 A100s 80GB, and all that code is in the main repo!
"""

#@title Single example
import json
import os
import subprocess
# import transformers
from transformers import PreTrainedModel


def inference_one(model, sequence, max_length):
    # create tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    #### Single embedding example ####

    tok_seq = tokenizer(sequence)
    tok_seq = tok_seq["input_ids"]  # grab ids

    # place on device, convert to tensor
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
    tok_seq = tok_seq.to(device)

    model.eval()

    # prep model and forward
    with torch.inference_mode():
        embeddings = model(tok_seq)
    return embeddings.mean(axis=1)


def inference_single():

    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # you only need to select which model to use here, we'll do the rest!
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = 16384  # auto selects

    # data settings:
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            '/home/projects/matrix/data/dna_transformers',
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
        model.to(device)

    # from scratch
    elif pretrained_model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    #### Single embedding example ####
    DIR = sys.argv[2]

    # df_class = pd.read_csv(os.path.join(DIR, '10000_5_families.csv'))
    # embeddings = []
    # species = []
    # genera = []
    # ids = []
    # families = []
    # df_path = pd.read_csv('/home/nmb127/code/data/genome_paths_map.csv')
    # PATHS_MAP = {o: f for o, f in zip(df_path['accession'], df_path['0'])}
    # for filename, df_g in df_class.groupby('original'):
    #     print(filename)
    #     seqs = {name: seq for name, seq in pyfastx.Fastx(os.path.join('/home/nmb127/code/data/', PATHS_MAP[filename]))}
    #     for i, row in df_g.iterrows():
    #         seq = seqs[row['assembly']]
    #         embs = []
    #         for start in range(0, len(seq), max_length):
    #             embs.append(inference_one(model, seq[start:start+max_length], max_length).cpu())
    #         embeddings.append(np.expand_dims(np.concatenate(embs).mean(axis=0), 0))
    #         species.append(row['original'])
    #         genera.append(row['g'])
    #         families.append(row['f'])
    #         ids.append(row['assembly'])
    # np.save(f'embeddings_5_families_{sys.argv[3]}.npy', np.concatenate(embeddings))
    # np.save(f'genera_5_families.npy', np.array(genera))
    # np.save(f'species_5_families.npy', np.array(species))
    # np.save(f'families_5_families.npy', np.array(families))
    # np.save(f'ids_5_families.npy', np.array(ids))

    # df_fams_set = pd.read_csv(os.path.join(DIR, 'marker_genes_eval.csv'))
    # embeddings = []
    # species = []
    # genes = []
    # families = []
    # FNA_PATH = '/home/nmb127/code/data/bac120_marker_genes_reps_r207/fna'
    # MAP_FILES = {
    #     'PF00380': 'PF00380.20',
    #     'PF00410': 'PF00410.20',
    #     'PF00466': 'PF00466.21',
    #     'PF01025': 'PF01025.20',
    #     'PF02576': 'PF02576.18',
    #     'PF03726': 'PF03726.15',
    # }
    # for g, df_g in df_fams_set.groupby('g'):
    #     print(g)
    #     seqs = {name: seq for name, seq in pyfastx.Fastx(os.path.join(FNA_PATH, f'{MAP_FILES.get(g, g)}.fna'))}
    #     for i, row in df_g.iterrows():
    #         seq = seqs[row['s']]
    #         embs = []
    #         for start in range(0, len(seq), max_length):
    #             embs.append(inference_one(model, seq[start:start+max_length], max_length).cpu())
    #         embeddings.append(np.expand_dims(np.concatenate(embs).mean(axis=0), 0))
    #         species.append(row['s'])
    #         genes.append(row['g'])
    #         families.append(row['f'])
    # np.save(f'embeddings_marker_genes_continue_{sys.argv[3]}.npy', np.concatenate(embeddings))
    # np.save(f'genes_marker_genes_continue.npy', np.array(genes))
    # np.save(f'species_marker_genes_continue.npy', np.array(species))
    # np.save(f'families_marker_genes_continue.npy', np.array(families))

    FILENAME = 'contigs_2kbp.fna.gz'
    seqs_all = {name: seq for name, seq in pyfastx.Fastx(os.path.join('/home/nmb127/code/data/', FILENAME))}
    df_fams_set = pd.read_csv(os.path.join(DIR, 'urog_contigs.csv'))
    embeddings = []
    species = []
    for i, row in df_fams_set.iterrows():
        if i % 100 == 0:
            print(i)
        seq = seqs_all[row['contigs']]
        embs = []
        for start in range(0, len(seq), max_length):
            embs.append(inference_one(model, seq[start:start+max_length], max_length).cpu())
        embeddings.append(np.expand_dims(np.concatenate(embs).mean(axis=0), 0))
        species.append(row['s'])
    np.save(f'embeddings_urog_finetuned_{sys.argv[3]}.npy', np.concatenate(embeddings))
    np.save(f'species_urog_finetuned.npy', np.array(species))

inference_single()


# to run this, just call:
    # python huggingface.py
