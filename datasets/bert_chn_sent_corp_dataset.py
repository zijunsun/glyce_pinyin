#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : bert_chn_sent_corp_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/20 20:31
@version: 1.0
@desc  : 
"""
import os
from functools import partial

import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader

from datasets.collate_functions import collate_to_max_length


class BertChnSentCorpDataset(Dataset):

    def __init__(self, directory, prefix, vocab_file, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        with open(os.path.join(directory, prefix + '.tsv'), 'r') as f:
            lines = f.readlines()
        self.lines = lines[1:]
        self.tokenizer = BertWordPieceTokenizer(vocab_file)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, sentence = line.split('\t', 1)
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=False).ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        input_ids = torch.LongTensor([101] + input_ids + [102])
        label = torch.LongTensor([int(label)])
        return input_ids, label


def unit_test():
    root_path = "/data/nfsdata2/sunzijun/glyce/tasks/ChnSentiCorp"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    prefix = "train"
    dataset = BertChnSentCorpDataset(directory=root_path, prefix=prefix, vocab_file=vocab_file)

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