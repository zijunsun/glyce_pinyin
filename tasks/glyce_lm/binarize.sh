export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
set RUST_BACKTRACE=1
USER_DIR="/home/sunzijun/glyce/preprocess_readers"
BERT_PATH="/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab"

READER_TYPE="glyce_pinyin_tokenize"

DATA_DIR="/data/nfsdata2/sunzijun/glyce/extract"
CONFIG_PATH="/data/nfsdata2/sunzijun/glyce/glyce/config"
LTP_DATA="/data/nfsdata2/nlp_application/models/ltp/ltp_data_v3.4.0"

DATA_BIN=${DATA_DIR}/bin
for phase in "task_data"; do
    INFILE=${DATA_DIR}/${phase}.txt;
    OFILE_PREFIX=${phase};
    shannon-preprocess \
        --input-file ${INFILE} \
        --output-file ${OFILE_PREFIX} \
        --destdir ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --reader-type ${READER_TYPE} \
        --bert_path $BERT_PATH \
        --config_path ${CONFIG_PATH} \
        --ltp_data ${LTP_DATA}  \
        --max_len 512 \
        --workers 1
#        --echo
done;
