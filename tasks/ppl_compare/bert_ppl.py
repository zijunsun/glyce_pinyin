#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : bert_ppl.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/21 17:57
@version: 1.0
@desc  : 
"""

import argparse
import os
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig, BertForMaskedLM

from datasets.collate_functions import collate_to_max_length
from datasets.static_glyce_dataset import StaticGlyceMaskLMDataset
from metrics.classification import MaskedAccuracy
from models.modeling_glycebert import GlyceBertForMaskedLM
from utils.radom_seed import set_random_seed

set_random_seed(0)


class BertPPL(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(args.bert_path)
        if self.args.mode == 'glyce':
            self.model = GlyceBertForMaskedLM.from_pretrained(self.bert_dir)
        else:
            self.model = BertForMaskedLM.from_pretrained(self.bert_dir)
        self.loss_fn = CrossEntropyLoss(reduction="none")
        self.acc = MaskedAccuracy(num_classes=self.bert_config.vocab_size)
        gpus_string = self.args.gpus if not self.args.gpus.endswith(',') else self.args.gpus[:-1]
        self.num_gpus = len(gpus_string.split(","))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    def forward(self, input_ids, pinyin_ids):
        """"""
        attention_mask = (input_ids != 0).long()
        if self.args.mode == "glyce":
            return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)
        else:
            return self.model(input_ids, attention_mask=attention_mask)

    def compute_loss_and_acc(self, batch):
        """"""
        epsilon = 1e-10
        input_ids, pinyin_ids, labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        masked_lms = labels.view(-1)
        outputs = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids)
        prediction_scores = outputs[0]
        label_mask = (masked_lms >= 0)
        # remove negative mask
        # masked_lms = torch.where(label_mask, masked_lms, torch.tensor(0, device=self.device, dtype=torch.int64))
        loss = self.loss_fn(prediction_scores.view(-1, self.bert_config.vocab_size), masked_lms)

        predict_labels = torch.argmax(prediction_scores.view(-1, self.bert_config.vocab_size), dim=-1)
        acc = self.acc(pred=predict_labels,
                       target=masked_lms,
                       mask=label_mask.long())

        label_mask = label_mask.float()
        loss *= label_mask
        loss = loss.sum() / (label_mask.sum() + epsilon)
        return loss, acc

    def test_dataloader(self):
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        return {'test_loss': loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {'test_loss': avg_loss, 'val_acc': avg_acc}
        print(avg_loss, avg_acc)
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        dataset = StaticGlyceMaskLMDataset(directory=self.args.data_dir,
                                           vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                                           prefix=prefix,
                                           max_length=self.args.max_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length, fill_values=[0, 0, -100]),
            drop_last=True
        )
        return dataloader


def main():
    """main"""
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--mode", default="bert", type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=0, type=int, help="save topk checkpoint")
    parser = BertPPL.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = BertPPL(args)
    trainer = Trainer.from_argparse_args(args, distributed_backend="ddp")
    trainer.test(model)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
