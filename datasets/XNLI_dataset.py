#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : XNLI_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/21 12:00
@version: 1.0
@desc  : 
"""

import json
import os
from functools import partial
from typing import List

import tokenizers
import torch
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader

from datasets.collate_functions import collate_to_max_length


class XNLIDataset(Dataset):

    def __init__(self, directory, prefix, vocab_file, config_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        with open(os.path.join(directory, 'xnli_' + prefix), 'r') as f:
            lines = f.readlines()
        self.lines = lines
        self.tokenizer = BertWordPieceTokenizer(vocab_file)

        # 加载拼音映射字典
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # 加载字符id映射tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # 加载拼音映射tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

        self.label_map = {"entailment": 0, "neutral": 1, "contradiction": 2, "contradictory": 2}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        first, second, third = line.strip().split('\t', 2)
        first_output = self.tokenizer.encode(first, add_special_tokens=False)
        first_pinyin_tokens = self.convert_sentence_to_pinyin_ids(first, first_output)
        second_output = self.tokenizer.encode(second, add_special_tokens=False)
        second_pinyin_tokens = self.convert_sentence_to_pinyin_ids(second, second_output)
        label = self.label_map[third]
        # 将句子转为ids
        bert_tokens = first_output.ids + [102] + second_output.ids
        pinyin_tokens = first_pinyin_tokens + [[0] * 8] + second_pinyin_tokens
        if len(bert_tokens) > self.max_length - 2:
            bert_tokens = bert_tokens[:self.max_length - 2]
            pinyin_tokens = pinyin_tokens[:self.max_length - 2]

        # 验证正确性，id个数应该相同
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # 转化list为tensor
        input_ids = torch.LongTensor([101] + bert_tokens + [102])
        pinyin_ids = torch.LongTensor([[0] * 8] + pinyin_tokens + [[0] * 8]).view(-1)
        label = torch.LongTensor([int(label)])
        return input_ids, pinyin_ids, label

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # 获取句子拼音
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # 获取有中文的位置的拼音
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # 不是中文字符，则跳过
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # 根据BPE的结果，取有中文的地方，构造ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids


def unit_test():
    root_path = "/data/nfsdata2/sunzijun/glyce/tasks/XNLI"
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    prefix = "train"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    dataset = XNLIDataset(directory=root_path, prefix=prefix, vocab_file=vocab_file,
                          config_path=config_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0])
    )
    for input_ids, pinyin_ids, label in dataloader:
        bs, length = input_ids.shape
        print(input_ids.shape)
        print(pinyin_ids.reshape(bs, length, -1).shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()
