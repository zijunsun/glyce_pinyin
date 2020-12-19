# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: utils
@time: 2020/8/4 10:17

"""

import json
import os
import torch
from pypinyin import pinyin, Style
from tqdm import tqdm


def count_params(conv):
    """
    count models params num
    conv(torch.nn.Module)
    """
    return sum(p.numel() for p in conv.parameters())


def channel_shuffle(x, groups: int):
    """
    channel shuffle，为了缓解group conv造成的同源问题
    Args:
        groups(int): 组数
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def build_pinyin_map(output_file):
    """
        function to build pinyin_map.json
    Args:
        output_file: path to save pinyin_map
    """
    chars = ["0", "1", "2", "3", "4", "5", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
             "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    char2idx = {}
    for idx, char in enumerate(chars):
        char2idx[char] = idx
    result = {"idx2char": chars, "char2idx": char2idx}
    with open(output_file, 'w+', encoding='utf-8') as fout:
        json.dump(result, fout, ensure_ascii=False)


def build_id2pinyin_map(vocab_file, output_path):
    """
        构造映射
        id2pinyin = {3:[[1,2,3,0,0,0,0,0][1,2,2,0,0,0,0,0]],
                     4:[[1,2,2,0,0,0,0,0]]}
        pinyin2tensor = {'wo3':[1,2,3,0,0,0,0,0],
                         'hao2':[7,8,10,2,0,0,0,0]}
    """
    # 加载词表
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
    # 加载拼音字母映射
    with open(os.path.join("/data/nfsdata2/sunzijun/models/models/config", 'pinyin_map.json')) as fin:
        pinyin_dict = json.load(fin)

    id2pinyin = {}
    pinyin2tensor = {}
    for idx, line in tqdm(enumerate(lines)):
        char = line[:-1]
        if char.startswith("##"):
            char = char[2:]
        if len(char) != 1:
            id2pinyin[idx] = [[0]*8]
        else:
            # 只处理字符长度为1的情况
            char_pinyins = pinyin(char, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])[0]
            result_ids = []
            for char_pinyin in char_pinyins:
                pinyin_string = char_pinyin
                ids = [0] * 8
                # 字符不是中文，直接置空
                if pinyin_string == "not chinese":
                    result_ids.append(ids)
                    break
                else:
                    # 如果最后一位，没有声调，则表示轻声，则加上5
                    if pinyin_string[-1] not in ["1", "2", "3", "4"]:
                        pinyin_string += '5'
                    for index, p in enumerate(pinyin_string):
                        if p not in pinyin_dict["char2idx"]:
                            break
                        ids[index] = pinyin_dict["char2idx"][p]
                    result_ids.append(ids)
                    pinyin2tensor[char_pinyin] = ids
            id2pinyin[idx] = result_ids

    with open(os.path.join(output_path, "id2pinyin.json"), 'w') as fout:
        json.dump(id2pinyin, fout, ensure_ascii=False)
    with open(os.path.join(output_path, "pinyin2tensor.json"), 'w') as fout:
        json.dump(pinyin2tensor, fout, ensure_ascii=False)


if __name__ =="__main__":
    vocab_file = "/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12/bert_chinese_base_large_vocab_20200704/vocab.txt"
    output_file = "/data/nfsdata2/sunzijun/models/models/config"
    build_id2pinyin_map(vocab_file, output_file)
