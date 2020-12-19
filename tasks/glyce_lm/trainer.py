#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/9/6 17:18
@version: 1.0
@desc  :
"""

import argparse
import json
import os
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from datasets.collate_functions import collate_to_max_length
from datasets.dynamic_glyce_dataset import DynamicGlyceMaskedLMDataset
from metrics.classification import MaskedAccuracy
from models.modeling_glycebert import GlyceBertForMaskedLM

from utils.radom_seed import set_random_seed

set_random_seed(0)


class GlyceBertLM(pl.LightningModule):
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
        self.model = GlyceBertForMaskedLM(self.bert_config)
        self.loss_fn = CrossEntropyLoss(reduction="none")
        self.acc = MaskedAccuracy(num_classes=self.bert_config.vocab_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, pinyin_ids, attention_mask=attention_mask)

    def compute_loss_and_acc(self, batch):
        """"""
        epsilon = 1e-10
        input_ids, pinyin_ids, labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        masked_lms = labels.view(-1)
        outputs = self.forward(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids
        )
        prediction_scores = outputs[0]
        label_mask = (masked_lms >= 0)
        # remove negative mask
        # masked_lms = torch.where(label_mask, masked_lms, torch.tensor(0, device=self.device, dtype=torch.int64))
        loss = self.loss_fn(prediction_scores.view(-1, self.bert_config.vocab_size),
                            masked_lms)

        predict_labels = torch.argmax(prediction_scores.view(-1, self.bert_config.vocab_size), dim=-1)
        acc = self.acc(pred=predict_labels,
                       target=masked_lms,
                       mask=label_mask.long())

        label_mask = label_mask.float()
        loss *= label_mask
        loss = loss.sum() / (label_mask.sum() + epsilon)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        loss, acc = self.compute_loss_and_acc(batch)
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(avg_loss, avg_acc)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        dataset = DynamicGlyceMaskedLMDataset(config_path=self.args.config_path,
                                              directory=self.args.data_dir,
                                              vocab_file=os.path.join(self.args.bert_path, "vocab.txt"),
                                              prefix=prefix,
                                              fields=None,
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
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--save_path", required=True, type=str, help="path to save checkpoint")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--config_path", required=True, type=str, help="config path")
    parser.add_argument("--save_topk", default=0, type=int, help="save topk checkpoint")
    parser = GlyceBertLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = GlyceBertLM(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=args.save_topk,
        save_last=True,
        verbose=True,
        monitor="train_loss",  # 考虑到eval可能有波动，但BERT/Roberta都report模型一直underfit
        period=-1,
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, 'checkpoint', "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         distributed_backend="ddp",
                                         logger=logger)

    trainer.fit(model)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
