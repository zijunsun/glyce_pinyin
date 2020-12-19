#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : pinyin.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/8/16 14:45
@version: 1.0
@desc  : 拼音的embedding
"""
import json
import os

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class PinyinEmbedding(nn.Module):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # 使用输入的pinyin_ids构造卷积输入
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]


def run_model():
    """run models"""

    # 初始化
    class Args:
        max_len = 64
        cws = True
        bert_path = "/data/nfsdata2/sunzijun/BertLM/roberta_3_epoch1_bin"
        input_file = "/data/nfsdata2/sunzijun/models/models/data/dev.txt"

    glyce_reader = GlyceTokenizeReader(Args)
    pinyin_emb = PinyinEmbedding(embedding_size=64, pinyin_out_dim=768)
    with open(Args.input_file, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        sentence = line.strip()
        tokenizer_output = glyce_reader.tokenizer.encode(sentence)
        pinyin_ids = glyce_reader.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)

        # 测试构造结果
        pinyin_ids_tensor = torch.tensor([pinyin_ids], dtype=torch.long)

        # print(pinyin_emb)
        # print('No. Parameters', count_params(pinyin_emb))
        y = pinyin_emb(pinyin_ids_tensor)
        # print(y.shape)


if __name__ == '__main__':
    run_model()
