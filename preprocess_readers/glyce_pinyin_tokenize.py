#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : glyce_pinyin_tokenize.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/4 20:00
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
from tqdm import tqdm

from preprocess_modules.ltp_parser import LtpModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@DatasetReader.register("glyce_pinyin_tokenize")
class GlycePinyinTokenizeReader(DatasetReader):
    """
    对 glyce pretrain 的数据进行
        1. 将多句pack到max_len
        2. bert tokenize
    """

    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)
        self.args = args
        self.config_path = args.config_path
        self.mask_prob = 0.15
        self.max_len = args.max_len
        self.bert_tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_path, "vocab.txt"))
        self.pack_sentence = ""
        self.unpack_prob = 0.1
        self.long = 0
        self.short = 0
        # 加载拼音映射字典
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # 加载字符id映射tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # 加载拼音映射tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)
        # 加载分词模型
        self.ltp_model = LtpModel(model_path=args.ltp_data, jieba_cut=False)

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument("--max_len", type=int, default=512)
        parser.add_argument("--bert_path", required=True, type=str)
        parser.add_argument("--config_path", required=True, type=str)
        parser.add_argument("--ltp_data", required=True, type=str)

    @property
    def fields2dtypes(self):
        """
        define numpy dtypes of each field.
        """
        dic = {
            "input_ids": np.uint16,  # 注意当int超过65500时候就不能用uint16了
            "pinyin_ids": np.uint16,
            "cws_ids": np.int16
        }
        return dic

    @property
    def fields2shapes(self):
        """ fields2shapes """
        return {
            "input_ids": (-1,),
            "pinyin_ids": (-1, 8),
            "cws_ids": (-1,),
        }

    def get_inputs(self, line: str) -> List[Dict[str, torch.Tensor]]:
        """get input from file, 返回的id没有添加开头和结束的SEP"""
        sentence = line.strip()
        sent_length = len(sentence)
        # 删除长度为1-5的句子
        # if 0 < sent_length < 5:
        #     raise ValueError("invalid line")
        # 超过max_len的句子，直接卡掉后部分
        if sent_length > self.max_len - 2:
            sentence = sentence[:self.max_len - 2]
        # 如果pack的句子比max_len长，则初始化，并报错
        if len(self.pack_sentence) > self.max_len - 2:
            self.pack_sentence = ""
            raise ValueError("unexpected packed sentence")
        # 长度为0表示换了段落，添加[SEP]
        if len(sentence) == 0:
            if len(self.pack_sentence) != 0 and not self.pack_sentence.endswith("[SEP]"):
                self.pack_sentence += '[SEP]'
            return []
        # 如果当前句子加之前的大于最大长度，则开始生成训练数据
        if len(self.pack_sentence) + len(sentence) > self.max_len - 2:
            while self.pack_sentence[-5:] == '[SEP]' and self.pack_sentence:
                self.pack_sentence = self.pack_sentence[:-5]
            if not self.pack_sentence:
                return []
            pack_sample = self.generate_train_data(self.pack_sentence)
            # 对于当前句子，以unpack_prob概率，直接生成样本
            if random.random() < self.unpack_prob:
                self.pack_sentence = ""
                unpack_sample = self.generate_train_data(sentence)
                self.long += 1
                self.short += 1
                return [pack_sample, unpack_sample]
            else:
                self.pack_sentence = sentence
                self.long += 1
                return [pack_sample]
        else:
            self.pack_sentence += sentence
            return []

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

    def generate_train_data(self, sentence):
        """生成训练数据"""
        # 将句子转为ids
        tokenizer_output = self.bert_tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # 分词
        bert_tokens_cws = []
        offsets2cws = self.ltp_model.offsets2cws_id(sentence=sentence)
        for offset_start, offset_end in tokenizer_output.offsets:
            if offset_start == offset_end == 0:  # [CLS]/[SEP]
                continue
            if sentence[offset_start: offset_end] == "[SEP]":
                cws_start = cws_end = -1
            else:
                cws_start = offsets2cws[offset_start]
                cws_end = offsets2cws[offset_end - 1]
            if cws_start != cws_end:
                # 错误，则重新初始化
                self.pack_sentence = ""
                assert cws_start == cws_end, f"{sentence},{str(offsets2cws)},{offset_start},{offset_end}, " \
                                             f"{sentence[offset_start: offset_end]}"
            bert_tokens_cws.append(cws_start)
        cws_tokens = [-1] + bert_tokens_cws + [-1]

        # 验证正确性，id个数应该相同
        if not len(bert_tokens) <= self.max_len:
            self.pack_sentence = ""
            assert len(bert_tokens) <= self.max_len
        if not len(bert_tokens) == len(pinyin_tokens) == len(cws_tokens):
            self.pack_sentence = ""
            assert len(bert_tokens) == len(pinyin_tokens) == len(cws_tokens)
        # 转化list为tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens)
        cws_ids = torch.LongTensor(cws_tokens)

        # 转化为sample
        sample = {"input_ids": input_ids,
                  "pinyin_ids": pinyin_ids,
                  "cws_ids": cws_ids}

        return sample

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
        max_len = 512
        cws = True
        bert_path = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab"
        # input_file = "/data/nfsdata2/sunzijun/glyce/glyce/data/dev.txt"
        input_file = "/data/nfsdata2/sunzijun/glyce/extract/task_data.txt"
        config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
        ltp_data = "/data/nfsdata2/nlp_application/models/ltp/ltp_data_v3.4.0"

    reader = GlycePinyinTokenizeReader(Args)
    with open(Args.input_file) as fin:
        for line in tqdm(fin):
            try:
                y = reader.get_inputs(line)
                if y:
                    print(y[0]['input_ids'].shape)
            except Exception as e:
                print(traceback.print_exc())
    print(reader)


if __name__ == '__main__':
    run_bert_tokenize_reader()
