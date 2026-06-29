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

import re
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def subm_tfidf_cosine(input_str: str, concept_units: list[str]) -> float:
    """
    基于TF-IDF余弦相似度的最大覆盖子模函数。
    
    Args:
        input_str (str): 当前的输入字符串。
        concept_units (List[str]): 概念单元列表。
        
    Returns:
        float: 总得分。
    """
    if input_str == '':
        # print("similarities", [0.0] * len(concept_units))
        return [0.0] * len(concept_units)
    
    # 建立文档集：input_str + concept_units
    documents = [input_str] + concept_units
    
    # TF-IDF编码
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 计算input与每个concept_unit的相似度
    input_vec = tfidf_matrix[0]  # 第一行是input
    concept_vecs = tfidf_matrix[1:]  # 后面是concepts
    
    # 两两计算余弦相似度
    similarities = cosine_similarity(input_vec, concept_vecs).flatten()
    
    # 只保留非负值，累加
    total_score = np.maximum(similarities, 0)
    # print("input_str", input_str)
    # print("concept_units", concept_units)
    # print("similarities", similarities)
    
    return total_score


def is_repeat_part(current_info, history_infos):
    # print("history_infos", history_infos)
    if history_infos == []:
        return False
    current_info = normalize_answer(extract_search(current_info))
    history_infos = [normalize_answer(extract_search(history_info)) for history_info in history_infos]
    if current_info == "":
        return True
    history_infos = [h if h.strip() else "None" for h in history_infos]
    # print("extracted_history_infos", history_infos)
    try:
        sim_scores = subm_tfidf_cosine(input_str=current_info, concept_units=history_infos)
    except ValueError as e:
        sim_scores = [1.0]
    sim_score = max(sim_scores)
    if sim_score > 0.8:
        return True
    else:
        return False

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
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def extract_search(search_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    search_pattern = r'<search>(.*?)</search>'
    match = re.finditer(search_pattern, search_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return " "
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def is_valid_search_part(part_text):
    """
    校验单个轨迹部分的格式，只判断 think 和 search 标签的逻辑嵌套。
    
    Args:
        part_text (str): 轨迹部分内容
    
    Returns:
        (bool, str): 是否有效，以及错误提示或成功信息
    """
    # 检查标签配对
    tags_to_check = ["think", "search"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", part_text))
        closing_count = len(re.findall(f"</{tag}>", part_text))
        if opening_count != closing_count:
            return False

    # 按标签拆分
    split_pattern = r"(</?(?:think|search)>)"
    parts = re.split(split_pattern, part_text)
    
    state = "start"  # 状态机: start -> in_think -> after_think -> in_search -> after_search -> ...
    
    for part in parts:
        if not part.strip():
            continue
        
        if re.match(r"</?(?:think|search)>", part):
            # 标签逻辑
            if part == "<think>" and state in ["start"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state in ["after_think"]:
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            else:
                return False
        else:
            # 标签内的内容允许存在
            if state not in ["in_think", "in_search"]:
                if part.strip():
                    return False

    # 最终必须以 </search> 结束，即 state == after_search
    if state != "after_search":
        return False
    
    return True

def is_valid_answer_part(part_text):
    """
    校验单个轨迹部分的格式，只判断 think 和 answer 标签的逻辑嵌套。
    
    Args:
        part_text (str): 轨迹部分内容
    
    Returns:
        (bool, str): 是否有效，以及错误提示或成功信息
    """
    # 检查标签配对
    tags_to_check = ["think", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", part_text))
        closing_count = len(re.findall(f"</{tag}>", part_text))
        if opening_count != closing_count:
            return False

    # 按标签拆分
    split_pattern = r"(</?(?:think|answer)>)"
    parts = re.split(split_pattern, part_text)
    
    state = "start"  # 状态机: start -> in_think -> after_think -> in_answer -> after_answer -> ...
    
    for part in parts:
        if not part.strip():
            continue
        
        if re.match(r"</?(?:think|answer)>", part):
            # 标签逻辑
            if part == "<think>" and state in ["start"]:
                state = "in_think"
            elif part == "</think>" and state in ["in_think"]:
                state = "after_think"
            elif part == "<answer>" and state in ["after_think"]:
                state = "in_answer"
            elif part == "</answer>" and state in ["in_answer"]:
                state = "after_answer"
            else:
                return False
        else:
            # 标签内的内容允许存在
            if state not in ["in_think", "in_answer"]:
                if part.strip():
                    return False

    # 最终必须以 </answer> 结束，即 state == after_answer
    if state != "after_answer":
        return False
    
    return True
def extract_content_from_information(info_block: str) -> list[str]:
    """
    给定 <information>...</information> 内部的文本，
    抽取每个 Doc 段的正文，返回 List[str]（每个 Doc 一条）。
    """
    info_block = info_block.strip()
    if not info_block:
        return []

    pattern = r'Doc\s*\d+\(Title:\s*"?([^")]+)"?\)\s*'
    matches = list(re.finditer(pattern, info_block))

    contents: list[str] = []
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(info_block)

        segment = info_block[start:end].strip()

        norm_title = title.strip('"').strip()
        if segment.lower().startswith(norm_title.lower()):
            segment = segment[len(norm_title):].lstrip(" \t:，。:.-")

        if segment:
            contents.append(segment)

    return contents


def extract_information_contents(per_turn_infos: list[str]) -> list[list[str]]:
    """
    对整个 per_turn_infos 列表：
    - 有 <information>...</information> 则返回该轮的 List[str]（按 Doc 分条）
    - 没有则返回空 list []
    最终返回 List[List[str]]，与轮数等长。
    """
    results: list[list[str]] = []
    for text in per_turn_infos:
        if not text:
            results.append([])
            continue

        m = re.search(r'<information>(.*?)</information>', text, re.DOTALL)
        if not m:
            results.append([])
        else:
            info_block = m.group(1)
            contents = extract_content_from_information(info_block)
            results.append(contents)
    return results

def step_information_gain(informations:list[list[str]], golden_info:list[dict])->list[float]:
    """
    Calculate the information gain for each information in the list.
    
    Args:
        informations: List of list of String
        golden_info: List of dicts with keys 'title' and 'paragraph_text'

    Returns:
        List of information gain scores
    """

    if len(golden_info) == 0:
        return [], []
    # Normalize golden info
    golden_infos = [normalize_answer(info['paragraph_text']) for info in golden_info]
    previous_match_degree = [0.0 for _ in golden_infos]

    # Calculate information gain for each information
    information_gains = []
    for information in informations:
        current_match_degree = [0.0 for _ in golden_infos]
        # Normalize information
        for info in information:
            info = normalize_answer(info)

            # Calculate information gain
            for i, score in enumerate(subm_tfidf_cosine(input_str=info, concept_units=golden_infos)):
                current_match_degree[i] = max(current_match_degree[i], score)

        information_gains.append(
            sum(max(current_match_degree[i] - previous_match_degree[i], 0) for i in range(len(current_match_degree)))
            / len(current_match_degree)
            )

        previous_match_degree = [max(current_match_degree[i], previous_match_degree[i]) for i in range(len(current_match_degree))]
    
    redundancy_penalty = [0.0 for _ in information_gains]
    info_gotten = set()
    for i, information in enumerate(informations):
        for info in information:
            if info in info_gotten:
                redundancy_penalty[i] += (1 / len(information))
                information_gains[i] = 0
            else:
                info_gotten.add(info)
    
    
    return information_gains, redundancy_penalty
# def step_information_gain(
#     informations: list[list[str]],
#     golden_info: list[dict]
# ) -> tuple[list[float], list[float]]:

#     if len(golden_info) == 0:
#         return [], []

#     # ---------- 预处理 golden infos ----------
#     golden_infos = [
#         normalize_answer(info["paragraph_text"])
#         for info in golden_info
#     ]
#     previous_match_degree = [0.0 for _ in golden_infos]

#     # ---------- 信息增益 ----------
#     information_gains: list[float] = []

#     for information in informations:
#         current_match_degree = [0.0 for _ in golden_infos]

#         for info in information:
#             norm_info = normalize_answer(info)
#             if not norm_info:
#                 continue

#             scores = subm_tfidf_cosine(
#                 input_str=norm_info,
#                 concept_units=golden_infos
#             )

#             # 累积当前轮对每个 golden 的最大匹配度
#             for i, score in enumerate(scores):
#                 if score > current_match_degree[i]:
#                     current_match_degree[i] = score

#         # 信息增益：相对前一轮匹配度的提升
#         gain = sum(
#             max(current_match_degree[i] - previous_match_degree[i], 0.0)
#             for i in range(len(current_match_degree))
#         ) / float(len(current_match_degree))
#         information_gains.append(gain)

#         # 更新历史最大匹配度
#         previous_match_degree = [
#             max(current_match_degree[i], previous_match_degree[i])
#             for i in range(len(current_match_degree))
#         ]

#     # ---------- 新信息奖励（基于 golden_infos + 阈值 0.8） ----------
#     # 规则：
#     # 1. 某一轮没有任何满足条件的信息 -> 该轮标记为 0
#     # 2. 对轮中每条 info：
#     #    - 归一化后 norm_info 非空；
#     #    - 对 golden_infos 计算 TF-IDF 余弦相似度；
#     #    - 若 max_score > 0.8 且 norm_info 之前从未作为“成功信息”出现过：
#     #         则该轮记为一次成功轮（只需记一次），并将 norm_info 加入 seen_infos。
#     # 3. 最终将 0.4 平均分配给所有成功轮：每个成功轮得到 0.4 / 成功轮数。
#     # 4. 若无成功轮，则所有轮奖励为 0。
#     # seen_infos: set[str] = set()
#     # success_flags = [0] * len(informations)

#     # for idx, information in enumerate(informations):
#     #     round_success = False
#     #     for info in information:
#     #         norm_info = normalize_answer(info)
#     #         if not norm_info or norm_info in seen_infos:
#     #             continue
#     #         scores = subm_tfidf_cosine(input_str=norm_info, concept_units=golden_infos)
#     #         if len(scores) == 0:
#     #             continue
#     #         max_score = float(np.max(scores))
#     #         if max_score > 0.8:  
#     #             round_success = True
#     #             seen_infos.add(norm_info)
#     #             break
#     #     success_flags[idx] = 1 if round_success else 0

#     # success_count = sum(success_flags)
#     # novelty_rewards = ([0.0] * len(informations) if success_count == 0
#     #                    else [0.4 / float(success_count) * f for f in success_flags])

#     # return information_gains, novelty_rewards
#     seen_infos: set[str] = set()
#     novelty_rewards = [0.0] * len(informations)
#     unit = 0.4 / float(len(golden_infos))  # 每次命中的加分

#     for idx, information in enumerate(informations):
#         for info in information:
#             norm_info = normalize_answer(info)
#             if not norm_info or norm_info in seen_infos:
#                 continue
#             scores = subm_tfidf_cosine(input_str=norm_info, concept_units=golden_infos)
#             if len(scores) == 0:
#                 continue
#             max_score = float(np.max(scores))
#             if max_score > 0.5:
#                 novelty_rewards[idx] += unit
#                 seen_infos.add(norm_info)

#     return information_gains, novelty_rewards

def compute_score_em(solution_str, ground_truth, new_parts, per_turn_infos, method='strict', support_docs=None, structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer_score = 0.0
    answer = extract_solution(solution_str=solution_str)

    step_scores = []
    step_valids = []
    for i, part in enumerate(new_parts):
        if i == len(new_parts) - 1:
            # 最后一个部分用 is_valid_answer_part 判断
            valid = is_valid_answer_part(part)
        else:
            valid = is_valid_search_part(part)
            repeat = is_repeat_part(part, new_parts[:i])
            valid = valid and not repeat
        # print(f"Part {i}: {part} -> valid={valid}")
        step_valids.append(1 if valid else 0) 
        step_scores.append(0)

    count_ones = sum(step_valids[:-1])
    count_zeros = len(step_valids) -1 - count_ones

    # 答案奖励
    if em_check(answer, ground_truth['target']):
        # answer_score = score-3*structure_format_score
        answer_score = score
    else:
        answer_score = 0
    
    # 最终回答格式奖励
    if step_valids[-1] == 1:
        step_scores[-1] = final_format_score

    else:
        step_scores[-1] = -0.5
    
    # 鼓励探索+防止重复
    for i in range(len(step_valids)-1):
        if step_valids[i] == 1:
            step_scores[i] = structure_format_score/8
        else:
            # step_scores[i] = 0
            step_scores[i] = -structure_format_score/8


    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"----------------rm_f1_steps_plan_with_support_docs----------------")
        print(f"Solution string: {solution_str}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Answer_score: {answer_score}")
        print(f"Step scores: {step_scores}")
        # while True:
        #     pass


    return {
        'answer_score': answer_score,
        'step_scores': step_scores
    }
 

