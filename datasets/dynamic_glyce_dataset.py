#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : dynamic_glyce_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/5 11:57
@version: 1.0
@desc  : 
"""
import json
import os
import random

import numpy as np
import torch
from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DynamicGlyceMaskedLMDataset(Dataset):
    """Dynamic Masked Language Model Dataset"""

    def __init__(self, config_path, directory, prefix, fields=None, vocab_file: str = "", mask_prob: float = 0.15,
                 max_length: int = 512, use_memory=False):
        super().__init__()
        fields = fields or ["input_ids", "pinyin_ids", "cws_ids"]

        self.fields2datasets = {}
        self.fields = fields
        self.mask_prob = mask_prob
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(os.path.dirname(vocab_file))
        # 加载字符id映射tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)

        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        input_ids = self.fields2datasets["input_ids"][item]
        pinyin_ids = self.fields2datasets["pinyin_ids"][item].view(-1, 8)
        cws_ids = self.fields2datasets["cws_ids"][item]

        # 90%概率使用whole word mask
        if random.random() < 0.9:
            masked_indices = self.whole_word_mask(input_ids, cws_ids)
        # 10%概率使用char word mask
        else:
            masked_indices = self.char_mask(input_ids)

        labels = input_ids.clone()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pinyin_ids[indices_replaced] = torch.LongTensor([0] * 8)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        indices_random_words = random_words[indices_random]

        replace_pinyin_tensor = self.convert_word_to_pinyin(indices_random_words)
        input_ids[indices_random] = indices_random_words
        pinyin_ids[indices_random] = replace_pinyin_tensor

        # pinyin 展开
        pinyin_ids = pinyin_ids.view(-1)

        return input_ids, pinyin_ids, labels

    def convert_word_to_pinyin(self, random_words: torch.Tensor):
        pinyin_ids = []
        for word in random_words:
            index = word.item()
            choice_pinyin = random.choice(self.id2pinyin[str(index)])
            pinyin_ids.append(choice_pinyin)
        if len(pinyin_ids) == 0:
            return torch.LongTensor([0] * 8)
        return torch.LongTensor(pinyin_ids)

    def whole_word_mask(self, input_ids: torch.Tensor, cws_ids: torch.Tensor) -> torch.Tensor:
        """
        whole word mask
        Args:
            input_ids: input ids [sent_len]
            cws_ids: char_offset to word_offset, [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        num_words = cws_ids.max().item() + 1
        num_mask = max(int(num_words * self.mask_prob), 1)
        mask_word_ids = np.random.choice(np.arange(num_words), size=num_mask, replace=False)
        masked_indices = torch.zeros_like(input_ids).bool()
        for mask_word_id in mask_word_ids:
            masked_indices = masked_indices | (cws_ids == mask_word_id)
        return masked_indices

    def char_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        random mask chars
        Args:
            input_ids: input ids [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                                     already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices


def run_zh():
    from tokenizers import BertWordPieceTokenizer

    data_path = "/data/nfsdata2/sunzijun/glyce/glyce/data/small_bin"
    bert_path = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"

    tokenizer = BertWordPieceTokenizer(os.path.join(bert_path, "vocab.txt"))
    prefix = "small"
    fields = None

    dataset = DynamicGlyceMaskedLMDataset(config_path=config_path, directory=data_path,
                                          vocab_file=os.path.join(bert_path, "vocab.txt"), prefix=prefix,
                                          fields=fields, max_length=512)
    print(len(dataset))
    from tqdm import tqdm
    for d in tqdm(dataset):
        print([v.shape for v in d])
        print(tokenizer.decode(d[0].tolist(), skip_special_tokens=False))
        tgt = [src if label == -100 else label for src, label in zip(d[0].tolist(), d[2].tolist())]
        print(tokenizer.decode(tgt, skip_special_tokens=False))


if __name__ == '__main__':
    run_zh()
