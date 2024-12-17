"""
In order to run this code you must prepare the ipts file.
The file must be saved as a JSONL file that contains dictionaries of the following form:
    {
        "idx": 0,
        "context": the entire context in a single string, 
    }
"""


import os
import json
import argparse
import jsonlines
from tqdm import tqdm
from multiprocessing import Pool
import re
import random
import sys

sys.path.append('../')
from utils.llm_api import query_llm


def parse_arguments():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="The LLM used to generate the questions.")
    parser.add_argument("--ipts", type=str, help="The path to the input sample saves as a JSONL file.")
    parser.add_argument("--api_key", type=str, help="The api key in case you are using closed models.")
    parser.add_argument("--out_dir", type=str, help="The output directory.")

    args = parser.parse_args()

    return args


args = parse_arguments()


MODEL = args.model
parallel_num = 1
ipts = [x for x in jsonlines.open(f"{args.ipts}")]
save_dir = f"{args.out_dir}"
os.makedirs(save_dir, exist_ok=True)
fout_path = f'{save_dir}/1_qa.jsonl'
if os.path.exists(fout_path):
    with jsonlines.open(fout_path, 'r') as f:
        opts = [x for x in f]
else:
    opts = []
s = set(x['idx'] for x in opts)
need_list = [x for x in ipts if x['idx'] not in s]
print(f'Already process: {len(opts)} | Remain to process: {len(need_list)}')


# def get_lang(input_text):
#     chinese_len = 0
#     english_len = 0

#     for word in input_text:
#         for char in word:
#             if '\u4e00' <= char <= '\u9fff':
#                 chinese_len += 1
#             elif '\u0041' <= char <= '\u007a':
#                 english_len += 1

#     if english_len > 4*chinese_len:
#         return "en"
#     else:
#         return "zh"


def get_lang(input_text):
    return "en"


# def generate_query(context):
#     lang = get_lang(context)
#     rd = random.random()
#     if lang == "zh":
#         if rd < 0.25:
#             prompt = context + '''\n\n请针对这篇文章，提出5个需要多段整合或总结才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#         if rd < 0.5:
#             prompt = context + '''\n\n请针对这篇文章，提出5个需要多跳推理才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#         elif rd < 0.75:
#             prompt = context + '''\n\n请针对这篇文章，提出5个中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#         else:
#             prompt = context + '''\n\n请针对这篇文章，提出5个信息查找类的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#     else:
#         if rd < 0.5:
#             rd = random.random()
#             if rd < 0.25:
#                 prompt = context + '''\n\n请针对这篇文章，提出5个需要多段整合或总结才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#             if rd < 0.5:
#                 prompt = context + '''\n\n请针对这篇文章，提出5个需要多跳推理才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#             elif rd < 0.75:
#                 prompt = context + '''\n\n请针对这篇文章，提出5个中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#             else:
#                 prompt = context + '''\n\n请针对这篇文章，提出5个信息查找类的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
#         else:
#             rd = random.random()
#             if rd < 0.25:
#                 prompt = context + '''\n\nGiven the above text, please propose 5 English questions that require summarization or integration from multiple parts, make sure they are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
#             if rd < 0.5:
#                 prompt = context + '''\n\nGiven the above text, please propose 5 English questions that require multi-hop reasoning, make sure they are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
#             elif rd < 0.75:
#                 prompt = context + '''\n\nGiven the above text, please propose 5 English questions that are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
#             else:
#                 prompt = context + '''\n\nGiven the above text, please propose 5 English information-seeking questions, make sure they are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
#     msg = [{'role': 'user', 'content': prompt}]
#     # print(prompt)
#     output = query_llm(msg, model=MODEL, temperature=0.0, max_new_tokens=2048, api_key=args.api_key)
#     # print(output)
#     if output is None:
#         return None
#     patterns = [r"1: (.*?)\n", r"2: (.*?)\n", r"3: (.*?)\n", r"4: (.*?)\n", r"5: (.*)"]
#     qs = []
#     for pattern in patterns:
#         match = re.search(pattern, output, re.DOTALL)
#         if match:
#             q = match.group(1).strip()
#             qs.append(q)
#     if len(qs) > 0:
#         q = random.sample(qs, 1)[0]
#         return q
#     else:
#         return None


def generate_query(context):
    rd = random.random()
    if rd < 0.25:
        prompt = context + '''\n\nGiven the above text, please propose 5 English questions that require summarization or integration from multiple parts, make sure they are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
    if rd < 0.5:
        prompt = context + '''\n\nGiven the above text, please propose 5 English questions that require multi-hop reasoning, make sure they are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
    elif rd < 0.75:
        prompt = context + '''\n\nGiven the above text, please propose 5 English questions that are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
    else:
        prompt = context + '''\n\nGiven the above text, please propose 5 English information-seeking questions, make sure they are diversed and cover all parts of the text, in the following format: "1: ", "2: ", ...'''
    msg = [{'role': 'user', 'content': prompt}]
    # print(prompt)
    output = query_llm(msg, model=MODEL, temperature=0.0, max_new_tokens=2048, api_key=args.api_key)
    # print("output: ", output)
    if output is None:
        return None
    patterns = [r"1: (.*?)\n", r"2: (.*?)\n", r"3: (.*?)\n", r"4: (.*?)\n", r"5: (.*)"]
    qs = []
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            q = match.group(1).strip()
            qs.append(q)
    # print("len(qs): ", len(qs))
    if len(qs) > 0:
        q = random.sample(qs, 1)[0]
        return q
    else:
        return None


def generate_answer(context, query):
    msg = [{'role': 'user', 'content': context + '\n\n' + query}]
    output = query_llm(msg, model=MODEL, temperature=1, max_new_tokens=2048, api_key=args.api_key)
    return output


def process(js):
    idx, context = js['idx'], js['context']
    query = generate_query(context)
    # print(query)
    if query is None:
        return 1
    answer = generate_answer(context, query)
    # print("answer", answer)
    if answer is None:
        return 1
    # print(answer)
    js = {
        'idx': idx,
        'query': query,
        'answer': answer,
        'context': context,
    }
    with open(fout_path, "a") as fout:
        fout.write(json.dumps(js, ensure_ascii=False)+'\n')
        fout.flush()
    return 0


# print("need_list: ", need_list)
with Pool(parallel_num) as p:
    rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))


num_bad_cases = sum(rst)
print(f'There are {num_bad_cases} bad cases. You can run this scripts again to re-process these bad cases.')
