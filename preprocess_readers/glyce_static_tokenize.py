#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : glyce_static_tokenize.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/22 10:41
@version: 1.0
@desc  : 
"""

import json
import os
import random
import traceback
from argparse import ArgumentParser
from typing import Dict
from typing import List

import numpy as np
import tokenizers
import torch
from pypinyin import pinyin, Style
from shannon_preprocessor.dataset_reader import DatasetReader
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

from utils.radom_seed import set_random_seed


@DatasetReader.register("glyce_static_tokenize")
class GlyceStaticTokenizdddeReader(DatasetReader):

    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)
        self.args = args
        self.config_path = args.config_path
        self.mask_prob = 0.15
        self.max_len = args.max_len
        self.bert_tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_path, "vocab.txt"))
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_path, "vocab.txt"))
        self.prev_tokens = []
        self.prev_pinyins = []
        set_random_seed(12)
        # 加载拼音映射字典
        with open(os.path.join(self.config_path, 'pinyin_map.json')) as fin:
            self.pinyin_dict = json.load(fin)
        # 加载字符id映射tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json')) as fin:
            self.id2pinyin = json.load(fin)
        # 加载拼音映射tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json')) as fin:
            self.pinyin2tensor = json.load(fin)

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument("--max_len", type=int, default=512)
        parser.add_argument("--bert_path", required=True, type=str)
        parser.add_argument("--config_path", required=True, type=str)

    @property
    def fields2dtypes(self):
        """
        define numpy dtypes of each field.
        """
        dic = {
            "input_ids": np.uint16,  # 注意当int超过65500时候就不能用uint16了
            "pinyin_ids": np.uint16,
            "labels": np.int16
        }
        return dic

    @property
    def fields2shapes(self):
        """ fields2shapes """
        return {
            "input_ids": (-1,),
            "pinyin_ids": (-1, 8),
            "labels": (-1,),
        }

    def get_inputs(self, line: str) -> List[Dict[str, torch.Tensor]]:
        """get input from file, 返回的id没有添加开头和结束的SEP"""
        sent = line.strip()
        output = []
        # 是空行，则向句子最后添加[SEP]
        if len(sent) == 0:
            if len(self.prev_tokens) != 0 and not self.prev_tokens[-1] == 102:
                self.prev_tokens.append(102)
                self.prev_pinyins.append([0] * 8)
            return []

        # 将句子转为ids
        tokenizer_output = self.bert_tokenizer.encode(sent, add_special_tokens=False)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sent, tokenizer_output)
        if len(bert_tokens) >= self.max_len - 2:
            self.max_len = self.args.max_len
            raise ValueError("line longer than max length")

        if len(bert_tokens) + len(self.prev_tokens) >= self.max_len - 2:
            # 足够写入的长度，开始mask和生成数据
            input_ids, pinyin_ids, labels = self.generate_train_data()
            output.append({"input_ids": input_ids,
                           "pinyin_ids": pinyin_ids,
                           "labels": labels})
            # 初始化缓存句子
            self.prev_tokens = bert_tokens
            self.prev_pinyins = pinyin_tokens
        else:
            self.prev_tokens += bert_tokens
            self.prev_pinyins.extend(pinyin_tokens)

        return output

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

        # 验证正确性，id个数应该相同
        assert len(pinyin_ids) == len(tokenizer_output.ids)
        return pinyin_ids

    def generate_train_data(self):
        """生成训练数据"""
        # 为数据添加[CLS] 和 [SEP]
        if self.prev_tokens[-1] != 102:
            self.prev_tokens = [101] + self.prev_tokens + [102]
            self.prev_pinyins = [[0] * 8] + self.prev_pinyins + [[0] * 8]
        else:
            self.prev_tokens = [101] + self.prev_tokens
            self.prev_pinyins = [[0] * 8] + self.prev_pinyins
        assert len(self.prev_tokens) <= self.max_len
        # 转化list为tensor
        input_ids = torch.LongTensor(self.prev_tokens)
        pinyin_ids = torch.LongTensor(self.prev_pinyins)

        # 90%概率使用whole word mask
        masked_indices = self.get_char_mask_indices(input_ids)

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

        return input_ids, pinyin_ids, labels

    def get_char_mask_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
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

    def convert_word_to_pinyin(self, random_words: torch.Tensor):
        pinyin_ids = []
        for word in random_words:
            index = word.item()
            choice_pinyin = random.choice(self.id2pinyin[str(index)])
            pinyin_ids.append(choice_pinyin)
        if len(pinyin_ids) == 0:
            return torch.LongTensor([0] * 8)
        return torch.LongTensor(pinyin_ids)


def run_bert_tokenize_reader():
    class Args:
        max_len = 256
        cws = True
        bert_path = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab"
        input_file = "/data/nfsdata2/sunzijun/glyce/glyce/data/dev.txt"
        config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"

    reader = GlyceStaticTokenizdddeReader(Args)
    with open(Args.input_file) as fin:
        for line in fin:
            try:
                y = reader.get_inputs(line)
                print(y)
            except Exception as e:
                print(traceback.print_exc())


if __name__ == '__main__':
    run_bert_tokenize_reader()
