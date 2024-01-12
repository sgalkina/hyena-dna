
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
            d_model=128, 
            n_layer=2, 
            d_inner=512, 
            vocab_size=12, 
            layer=dict(l_max=2050, emb_dim=5),
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

    # prep model and forward
    with torch.inference_mode():
        embeddings = model(tok_seq)
    return embeddings.mean(axis=1)


def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    """Training loop."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(data.size(), target.size())
        optimizer.zero_grad()
        print(model)
        output = model(data)
        print(output.size(), target.squeeze().size())
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def run_train():

    '''
    Main entry point for training.  Select the dataset name and metadata, as
    well as model and training args, and you're off to the genomic races!

    ### GenomicBenchmarks Metadata
    # there are 8 datasets in this suite, choose 1 at a time, with their corresponding settings
    # name                                num_seqs        num_classes     median len    std
    # dummy_mouse_enhancers_ensembl       1210            2               2381          984.4
    # demo_coding_vs_intergenomic_seqs    100_000         2               200           0
    # demo_human_or_worm                  100_000         2               200           0
    # human_enhancers_cohn                27791           2               500           0
    # human_enhancers_ensembl             154842          2               269           122.6
    # human_ensembl_regulatory            289061          3               401           184.3
    # human_nontata_promoters             36131           2               251           0
    # human_ocr_ensembl                   174756          2               315           108.1

    '''
    num_epochs = 100  # ~100 seems fine
    max_length = 500  # max len of sequence of dataset (of what you want)
    use_padding = True
    dataset_name = 'human_enhancers_cohn'
    batch_size = 256
    learning_rate = 6e-4  # good default for Hyena
    rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1

    # we need these for the decoder head, if using
    use_head = True
    n_classes = 1023

    backbone_cfg = None

    # you only need to select which model to use here, we'll do the rest!
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 2048,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = max_lengths[pretrained_model_name]  # auto selects

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            '/home/projects/matrix/data/dna_transformers',
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
        model.to(device)
    # from scratch
    else:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    # create datasets
    DIR = '/home/projects/matrix/data/dna_transformers/data'
    df_class = pd.read_csv(os.path.join(DIR, '1740_classes.csv'))
    df_seq = pd.read_csv(os.path.join(DIR, '1740_sequences.csv'))
    df = pd.merge(df_class, df_seq, left_on='assembly', right_on='id')
    dataset = CustomDataset(df)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)

    for epoch in range(num_epochs):
        print(epoch)
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        optimizer.step()

    embeddings = []
    for i, seq in enumerate(df['sequence']):
        print(i)
        embeddings.append(inference_one(model, seq, max_length))

    np.save('embedding_finetuned_toy.npy', np.concatenate(embeddings))

    print(np.concatenate(embeddings).shape)  # embeddings here!

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe = dataframe.reset_index(drop=True)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        max_length = 1024
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )
        tok_seq = tokenizer(
            row['sequence'],
            add_special_tokens=False,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        tok_seq = tok_seq["input_ids"]
        seq = torch.LongTensor(tok_seq)
        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos
        return data, target

    def __len__(self):
        return len(self.dataframe)


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

    max_length = max_lengths[pretrained_model_name]  # auto selects

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
    # DIR = '/home/projects/matrix/data/dna_transformers/data'
    DIR = sys.argv[2]
    df_class = pd.read_csv(os.path.join(DIR, '1740_classes.csv'))
    df_seq = pd.read_csv(os.path.join(DIR, '1740_sequences.csv'))
    df = pd.merge(df_class, df_seq, left_on='assembly', right_on='id')
    embeddings = []
    for i, seq in enumerate(df['sequence']):
        if i % 10 == 0:
            print(i)
        embs = []
        for start in range(0, len(seq), max_length):
            embs.append(inference_one(model, seq[start:start+max_length], max_length).cpu())
        embeddings.append(np.expand_dims(np.concatenate(embs).mean(axis=0), 0))
    np.save('embeddings_fresh.npy', np.concatenate(embeddings))

# # uncomment to run! (to get embeddings)
inference_single()


# to run this, just call:
    # python huggingface.py
