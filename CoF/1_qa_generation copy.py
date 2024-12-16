from utils.llm_api import query_llm

import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def parse_arguments():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="The LLM used to generate the questions.")
    parser.add_argument("--ipts", type=str, help="The path to the input sample saves as a JSONL file.")
    parser.add_argument("--out_dir", type=str, help="The output directory.")

    args = parser.parse_args()

    return args


def load_jsonl_file(
    filepath: str
) -> List[Dict[str, Any]]:
    """
    Load a jsonl file into a list

    *arguments*
        *filepath* path to the file
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def make_dir(
    path: str
) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_lang(
    input_text: str
) -> str:
    """
    Distinguishes between Chinese and English.

    *args*
        *input_text*: the input string
    """
    chinese_len = 0
    english_len = 0

    for word in input_text:
        for char in word:
            if '\u4e00' <= char <= '\u9fff':
                chinese_len += 1
            elif '\u0041' <= char <= '\u007a':
                english_len += 1

    if english_len > 4*chinese_len:
        return "en"
    else:
        return "zh"


def generate_query(
    context: str,
    MODEL: str,
):
    """
    Generates queries from the context.

    *args*
        *context*: the context in string format.
        *MODEL*: the chosen model.
    """
    lang = get_lang(context)
    rd = random.random()
    if lang == "zh":
        if rd < 0.25:
            prompt = context + '''\n\n请针对这篇文章，提出5个需要多段整合或总结才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
        if rd < 0.5:
            prompt = context + '''\n\n请针对这篇文章，提出5个需要多跳推理才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
        elif rd < 0.75:
            prompt = context + '''\n\n请针对这篇文章，提出5个中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
        else:
            prompt = context + '''\n\n请针对这篇文章，提出5个信息查找类的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
    else:
        if rd < 0.5:
            rd = random.random()
            if rd < 0.25:
                prompt = context + '''\n\n请针对这篇文章，提出5个需要多段整合或总结才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
            if rd < 0.5:
                prompt = context + '''\n\n请针对这篇文章，提出5个需要多跳推理才能回答的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
            elif rd < 0.75:
                prompt = context + '''\n\n请针对这篇文章，提出5个中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
            else:
                prompt = context + '''\n\n请针对这篇文章，提出5个信息查找类的中文问题，保证问题多样性、尽可能覆盖全部内容，格式如下: "1: ", "2: ", ...'''
        else:
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
    output = query_llm(msg, model=MODEL, temperature=1, max_new_tokens=2048)
    # print(output)
    if output is None:
        return None
    patterns = [r"1: (.*?)\n", r"2: (.*?)\n", r"3: (.*?)\n", r"4: (.*?)\n", r"5: (.*)"]
    qs = []
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            q = match.group(1).strip()
            qs.append(q)
    if len(qs) > 0:
        q = random.sample(qs, 1)[0]
        return q
    else:
        return None


def generate_answer(
    context: str,
    query: str,
    MODEL: str
) -> str:
    """
    Generates the answer starting from the query and the context.

    *args*
        *context*: the string representing the context.
        *query*: the string representing the query.
        *MODEL*: the chosen model.
    """

    msg = [{'role': 'user', 'content': context + '\n\n' + query}]
    output = query_llm(msg, model=MODEL, temperature=1, max_new_tokens=2048)
    return output


def process(
    js: Dict[str, Any],
    fout_path: str,
) -> Dict[str, Any]:
    idx, context = js['idx'], js['context']
    query = generate_query(context)
    # print(query)
    if query is None:
        return 1
    answer = generate_answer(context, query)
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


def main():
    args = parse_arguments()
    MODEL = args.model
    OUT_DIR = args.out_dir
    make_dir(OUT_DIR)

    ipts = load_jsonl_file(args.ipts)  # inputs
    opts = load_jsonl_file(f"{OUT_DIR}/1_qa.jsonl")  # already generated output
    s = set(x['idx'] for x in opts)
    need_list = [x for x in ipts if x['idx'] not in s]

    print(f'Already process: {len(opts)} | Remain to process: {len(need_list)}')

    rst = list(map(process, need_list))
    num_bad_cases = sum(rst)

    print(f'There are {num_bad_cases} bad cases. You can run this scripts again to re-process these bad cases.')


if __name__ == "__main__":
    main()
