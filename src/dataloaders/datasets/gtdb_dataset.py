
from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import pyfastx
import os


"""

Dataset for sampling arbitrary intervals from the human genome.

"""

# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        # max_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        pad_interval = False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = {name: seq for name, seq in pyfastx.Fastx(str(fasta_file))}

        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval        

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            # remove tail end, might be gibberish code
            # truncate_len = int(len(self.seqs[chr_name]) * 0.9)
            # self.chr_lens[chr_name] = truncate_len
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, start, end, max_length, return_augs = False):
        """
        max_length passed from dataset, not from init
        """
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        # chromosome_length = len(chromosome)
        chromosome_length = self.chr_lens[chr_name]

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        # checks if not enough sequence to fill up the start to end
        if interval_length < max_length:
            extra_seq = max_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        # Added support!  need to allow shorter seqs
        if interval_length > max_length:
            end = start + max_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq

class GTDBDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''

    def __init__(
        self,
        split,
        species,
        fasta_paths,
        fasta_root,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        task='next_token',
    ):

        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval
        SPLITS_CONFIG = {
            'train': 1,
            'test': 1,
            'valid': 1,
        }
        self.N_split = SPLITS_CONFIG[split]
        self.split = split
        self.return_seq_indices = return_seq_indices
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

        self.files = {}
        self.fasta_paths = fasta_paths
        self.fasta_root = fasta_root
        with open(self.fasta_paths) as f:
            self.filenames = [os.path.join(fasta_root, l.strip()) for l in f]
        self.species = list(set(self.filenames))
        print(f'Number of species (files) is {len(self.species)} ({self.species[:3]}...)')

        self.task = task
        print(f'Task is {self.task}')

        self._index_files()
        self.datasets = {
            'train': self.df,
            'test': self.df_test,
            'valid': self.df_test,
        }
        print(f'The dataset has {len(self.datasets["train"])} samples from {len(self.datasets["train"][0].unique())} files for training')
        print(f'The dataset has {len(self.datasets["test"])} samples from {len(self.datasets["test"][0].unique())} files for testing')

    def _index_files(self):
        """Like bed file for HG38 but generated on the fly"""
        # filenames, seq_names, starts, ends = [], [], [], []
        # filenames_test, seq_names_test, starts_test, ends_test = [], [], [], []
        filenames, seq_names, seqs = [], [], []
        filenames_test, seq_names_test, seqs_test = [], [], []
        for fasta_filename in self.filenames:
            fastafile = FastaInterval(
                fasta_file = fasta_filename,
                # max_length = max_length,
                return_seq_indices = self.return_seq_indices,
                shift_augs = self.shift_augs,
                rc_aug = self.rc_aug,
                pad_interval = self.pad_interval,
            )
            current_long = 0
            current_short = 0
            current_test = 0
            N_LONG = 100
            N_SHORT = 40
            N_TEST = 2
            for name, seq in fastafile.seqs.items():
                L = len(seq)
                if L <= self.max_length:
                    rand_start = 0
                    rand_end = L
                    if coin_flip():
                        if current_test < N_TEST:
                            filenames_test.append(fasta_filename)
                            seq_names_test.append(name)
                            seq = fastafile(name, rand_start, rand_end, max_length=self.max_length, return_augs=self.return_augs)
                            seqs_test.append(seq)
                            current_test += 1
                    else:
                        if current_short < N_SHORT:
                            filenames.append(fasta_filename)
                            seq_names.append(name)
                            seq = fastafile(name, rand_start, rand_end, max_length=self.max_length, return_augs=self.return_augs)
                            seqs.append(seq)
                            current_short += 1
                else:
                    # get around 2 samples from the long sequences for each species
                    N_draws = min(int(L / self.max_length), N_LONG - current_long)
                    for _ in range(N_draws):
                        rand_start = randrange(0, L - self.max_length)
                        rand_end = rand_start + self.max_length
                        filenames.append(fasta_filename)
                        seq_names.append(name)
                        seq = fastafile(name, rand_start, rand_end, max_length=self.max_length, return_augs=self.return_augs)
                        seqs.append(seq)
                        current_long += 1
        self.df = pd.DataFrame({0: filenames, 1: seq_names, 2: seqs})
        self.df = self.df.sample(frac=1) # shuffle
        self.df_test = pd.DataFrame({0: filenames_test, 1: seq_names_test, 2: seqs_test})
        self.df_test = self.df_test.sample(frac=1) # shuffle

    def __len__(self):
        return len(self.datasets[self.split])

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        df = self.datasets[self.split]
        row = df.iloc[idx]
        # row = (chr, start, end, split)
        filename, chr_name, seq = (row[0], row[1], row[2])

        # seq = self.files[filename](chr_name, start, end, max_length=self.max_length, return_augs=self.return_augs)

        if self.tokenizer_name == 'char':

            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        if self.task == 'next_token':
            target = seq[1:].clone()  # offset by 1, includes eos
        elif self.task == 'species_classification':
            target = self.species.index(filename)
        else:
            raise AttributeError('Unknown task, must be one of the ["next_token", "species_classification"]')

        return data, target
