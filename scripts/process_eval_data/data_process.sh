



#!/bin/bash

WORK_DIR=/data/home/zdhs0010/chongwenyue/Search-R1
LOCAL_DIR=$WORK_DIR/data/eval_data

# 定义要遍历的数据集列表
DATASETS="nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle"

# 循环处理每个数据集
for DATA in $DATASETS; do
    echo "Processing dataset: $DATA"
    python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
        --local_dir $LOCAL_DIR/$DATA \
        --data_sources $DATA
done

echo "All datasets processed!"





