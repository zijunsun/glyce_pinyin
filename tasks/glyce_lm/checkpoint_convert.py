#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : checkpoint_convert.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/20 19:07
@version: 1.0
@desc  : 
"""
import argparse
import os

import torch
from pytorch_lightning import Trainer

from tasks.glyce_lm.trainer import GlyceBertLM


def convert_checkpint_to_bin(checkpoint_path, bin_path, mode="cpu"):
    """将模型save的checkpoint转化为huggingface的bin文件"""
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--config_path", required=True, type=str, help="config path")

    parser = GlyceBertLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = GlyceBertLM(args)

    checkpoint = torch.load(checkpoint_path, map_location=mode)
    model.load_state_dict(checkpoint['state_dict'])
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)
    model.model.save_pretrained(bin_path)


if __name__ == "__main__":
    checkpoint_path = "/data/nfsdata2/sunzijun/glyce/glyce/checkpoint/epoch=5-val_loss=1.7038-val_acc=5.1195.ckpt"
    bin_path = "/data/nfsdata2/sunzijun/glyce/glyce_bert"
    convert_checkpint_to_bin(checkpoint_path, bin_path)
