#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : static_glyce_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/22 11:21
@version: 1.0
@desc  : 
"""
import os

from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from transformers import BertTokenizer


class StaticGlyceMaskLMDataset(Dataset):
    """Dynamic Masked Language Model Dataset"""

    def __init__(self, directory, prefix, fields=None, vocab_file: str = "", max_length: int = 128, use_memory=False):
        super().__init__()
        fields = fields or ["input_ids", "pinyin_ids", "labels"]
        self.fields2datasets = {}
        self.fields = fields
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.dirname(vocab_file))
        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        input_ids = self.fields2datasets["input_ids"][item]
        pinyin_ids = self.fields2datasets["pinyin_ids"][item]
        labels = self.fields2datasets['labels'][item]

        return input_ids, pinyin_ids, labels


def run():
    data_path = "/data/nfsdata2/sunzijun/glyce/glyce/data/bin"
    bert_path = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab"

    tokenizer = BertWordPieceTokenizer(os.path.join(bert_path, "vocab.txt"))
    prefix = "dev"
    dataset = StaticGlyceMaskLMDataset(data_path, vocab_file=os.path.join(bert_path, "vocab.txt"),
                                       prefix=prefix, max_length=512)
    print(len(dataset))
    from tqdm import tqdm
    for d in tqdm(dataset):
        print([v.shape for v in d])
        print(tokenizer.decode(d[0].tolist(), skip_special_tokens=False))
        tgt = [src if label == -100 else label for src, label in zip(d[0].tolist(), d[2].tolist())]
        print(tokenizer.decode(tgt, skip_special_tokens=False))


if __name__ == '__main__':
    run()
