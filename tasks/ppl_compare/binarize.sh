export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

USER_DIR="/home/sunzijun/glyce/preprocess_readers"
#BERT_PATH="/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab"
#BERT_PATH="/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12"
BERT_PATH="/data/nfsdata2/nlp_application/models/bert/bert_xunfei/chinese_roberta_wwm_ext_pytorch"
CONFIG_PATH="/data/nfsdata2/sunzijun/glyce/glyce/config"
READER_TYPE="glyce_static_tokenize"

DATA_DIR="/data/nfsdata2/sunzijun/glyce/glyce/evaluate/eval_roberta"


DATA_BIN=${DATA_DIR}/bin
for phase in 'test'; do
    INFILE=${DATA_DIR}/${phase}.txt;
    OFILE_PREFIX=${phase};
    shannon-preprocess \
        --input-file ${INFILE} \
        --output-file ${OFILE_PREFIX} \
        --destdir ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --reader-type ${READER_TYPE} \
        --bert_path $BERT_PATH \
        --config_path $CONFIG_PATH \
        --max_len 512 \
        --workers 16 \
        --echo
done;