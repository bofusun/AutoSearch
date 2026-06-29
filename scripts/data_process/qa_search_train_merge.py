# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
#         prefix = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> you are thinking or every time you get new information. \
The question may require you to search for multiple points separately. Please think about which questions or keys that need to be searched and list the questions or keys you plan to search. \
After that, you can search for one question by calling a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want but can only search one question at a time. You might need to rethink and adjust your questions list that you plan to search based on results. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>. The Question you need to answer: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
