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
from utils import make_prefix
import requests

##### utils

import json
import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            return 1
    return 0

def split_documents(text):
    # Find all the split points
    split_points = [m.start() for m in re.finditer(r'Doc \d+\(Title: .*?\)', text)]
    
    docs = []
    for i in range(len(split_points)):
        start = split_points[i]
        end = split_points[i+1] if i+1 < len(split_points) else len(text)
        doc_text = text[start:end]
        # Remove the "Doc {n}" prefix and just keep (Title...) onward
        cleaned_doc = re.sub(r'Doc \d+', '', doc_text).strip()
        docs.append(cleaned_doc)
    return docs

def subem_score(doc_str, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    doc_list = split_documents(doc_str)
    if 'no' in golden_answers or 'yes' in golden_answers:
        return random.random()
    score = 0.0
    for idx, doc in enumerate(doc_list):
        if subem_check(doc, golden_answers):
            score += 1 / (idx+1)
    return score


def subem_score_2(doc_str, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    doc_list = split_documents(doc_str)
    score = 0.0
    if 'no' in golden_answers or 'yes' in golden_answers:
        score += 1 - 1/len(doc_list) + random.random() / len(doc_list)
    else:
        for idx, doc in enumerate(doc_list):
            if subem_check(doc, golden_answers):
                score += 1 - idx / len(doc_list)
    if score > 0:
        score += random.random() / len(doc_list)
    return score

def subem_score_3(str_question, doc_str, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    doc_list = split_documents(doc_str)
    score = 0.0

    if 'no' in golden_answers or 'yes' in golden_answers:
        score += 1 - 1/len(doc_list)
    else:
        if subem_check(str_question, golden_answers):
            score += 1.0
        for idx, doc in enumerate(doc_list):
            if subem_check(doc, golden_answers):
                score += 1 - idx / len(doc_list)
    if score > 0:
        score += random.random() / len(doc_list)
    return score

def compute_filter_score(str_question, doc_str, golden_answers):
    subem_score = subem_score_3(str_question, doc_str, golden_answers)
    if subem_score > 1:
        return random.choice([2, 4, 6, 8, 10])
    else:
        return random.choice([1, 3, 5, 7, 9])

#####

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


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
    cache_file_path = os.path.join(args.local_dir, f'search.json')
    if not os.path.exists(cache_file_path):
        cache_data = {}
    else:
        with open(cache_file_path, 'r') as f:
            cache_data = json.load(f)

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']
        # train_dataset = train_dataset.shuffle(seed=42).select(range(100))

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

                str_question = example['question']
                if str_question in cache_data:
                    doc_str = cache_data[str_question]
                else:
                    doc_str = search(str_question)
                    cache_data[str_question] = doc_str
                    if random.randint(0, 100) ==0: # save every 100 times
                        with open(cache_file_path, 'w') as f:
                            json.dump(cache_data, f)
                filter_score = compute_filter_score(str_question, doc_str, example['golden_answers'])
                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "filter_score": filter_score,
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
        with open(cache_file_path, 'w') as f:
            json.dump(cache_data, f)
    

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset = all_train_dataset.sort('filter_score', reverse=True)
    # all_train_dataset = all_train_dataset.filter(lambda example: example['filter_score'] != 0)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    sorted_scores=[]
    for i in range(1000):
        sorted_scores.append(all_train_dataset[int(i*len(all_train_dataset)/1000)]['filter_score'])
    print(sorted_scores)
    with open(os.path.join(local_dir, 'train_sort.txt'), 'w') as f:
        for score in sorted_scores:
            f.write(f'{score}\n')

    assert hdfs_dir is None
