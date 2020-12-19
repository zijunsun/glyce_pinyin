python trainer.py \
--bert_path /data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab \
--data_dir /data/nfsdata2/sunzijun/glyce/tasks/ChnSentiCorp \
--save_path /data/nfsdata2/sunzijun/glyce/tasks/ChnSentiCorp \
--config_path /data/nfsdata2/sunzijun/glyce/glyce/config \
--pretrain_checkpoint /data/nfsdata2/sunzijun/glyce/glyce/checkpoint/epoch=0-val_loss=1.4566-val_acc=5.4527.ckpt \
--task ChnSentCorp \
--gpus=3,