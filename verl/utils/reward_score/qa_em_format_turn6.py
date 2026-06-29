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
        print("similarities", [0.0] * len(concept_units))
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
    print("similarities", similarities)
    
    return total_score

# def subm_tfidf_cosine(input_str: str, concept_units: list[str]) -> float:
#     """
#     基于TF-IDF余弦相似度的最大覆盖子模函数。
#     最小改动：空文档或空词汇表时直接返回全0。
#     """
#     if input_str == '':
#         return [0.0] * len(concept_units)

#     documents = [input_str] + concept_units
#     # 过滤空字符串，防止 sklearn 炸
#     documents = [d for d in documents if d.strip()]

#     # 全空就提前返回
#     if not documents:
#         return [0.0] * len(concept_units)

#     try:
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform(documents)
#     except ValueError as e:
#         # 空词汇表（全是停用词）也会走这里
#         if "empty vocabulary" in str(e):
#             return [0.0] * len(concept_units)
#         raise  # 其它错误继续抛

#     input_vec = tfidf_matrix[0]
#     concept_vecs = tfidf_matrix[1:]
#     similarities = cosine_similarity(input_vec, concept_vecs).flatten()
#     print("similarities", similarities)
#     return np.maximum(similarities, 0).tolist()

def is_repeat_part(current_info, history_infos):
    print("history_infos", history_infos)
    if history_infos == []:
        return False
    current_info = normalize_answer(extract_search(current_info))
    history_infos = [normalize_answer(extract_search(history_info)) for history_info in history_infos]
    if current_info == "":
        return True
    history_infos = [h if h.strip() else "None" for h in history_infos]
    # print("extracted_history_infos", history_infos)
    sim_scores = subm_tfidf_cosine(input_str=current_info, concept_units=history_infos)
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

def compute_score_em(solution_str, ground_truth, new_parts, per_turn_infos, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
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
        print(f"Part {i}: {part} -> valid={valid}")
        step_valids.append(1 if valid else 0) 
        step_scores.append(0)

    count_ones = sum(step_valids[:-1])
    count_zeros = len(step_valids) -1 - count_ones

    # 答案奖励
    if em_check(answer, ground_truth['target']):
        answer_score = score-structure_format_score
    else:
        answer_score = 0
    
    # 最终回答格式奖励
    if step_valids[-1] == 1:
        step_scores[-1] = final_format_score
    else:
        step_scores[-1] = 0
    
    # 鼓励探索+防止重复
    for i in range(len(step_valids)-1):
        if step_valids[i] == 1:
            step_scores[i] = structure_format_score / count_ones if count_ones>0 else 0
        else:
            step_scores[i] = -structure_format_score / count_zeros if count_zeros>0 else 0

    return {
        'answer_score': answer_score,
        'step_scores': step_scores,
    }
 

