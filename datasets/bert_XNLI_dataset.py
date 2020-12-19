#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : bert_XNLI_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/21 11:27
@version: 1.0
@desc  : 
"""
import os
from functools import partial

import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader

from datasets.collate_functions import collate_to_max_length


class BertXNLIDataset(Dataset):

    def __init__(self, directory, prefix, vocab_file, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        with open(os.path.join(directory, 'xnli_' + prefix), 'r') as f:
            lines = f.readlines()
        self.lines = lines
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2, "contradictory":2}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        first, second, third = line.strip().split('\t', 2)
        first_input_ids = self.tokenizer.encode(first, add_special_tokens=False).ids
        second_input_ids = self.tokenizer.encode(second, add_special_tokens=False).ids
        label = self.label_map[third]
        input_ids = first_input_ids + [103] + second_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        input_ids = torch.LongTensor([101] + input_ids + [102])
        label = torch.LongTensor([int(label)])
        return input_ids, label


def unit_test():
    root_path = "/data/nfsdata2/sunzijun/glyce/tasks/XNLI"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    prefix = "train"
    dataset = BertXNLIDataset(directory=root_path, prefix=prefix, vocab_file=vocab_file)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0])
    )
    for input_ids, label in dataloader:
        print(input_ids.shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()
