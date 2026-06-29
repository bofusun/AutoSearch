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
            score = 0.8
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
            if part == "<think>" and state == "start":
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state in ["start", "after_think"]:
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
            if part == "<think>" and state == "start":
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<answer>" and state in ["start", "after_think"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
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

def compute_score_em(solution_str, ground_truth, new_parts, method='strict', format_score=0., score=1.0):
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
    for i, part in enumerate(new_parts):
        if i == len(new_parts) - 1:
            # 最后一个部分用 is_valid_answer_part 判断
            valid = is_valid_answer_part(part)
        elif i % 2 == 0:
            # 偶数项（从0开始计数）用 is_valid_search_part 判断
            valid = is_valid_search_part(part)
        else:
            # 奇数项直接认为无效（或根据需要处理）
            valid = False
        print(f"Part {i}: {part} -> valid={valid}")
        step_scores.append(1 if valid else 0) 
    # 先统计偶数项中为1和为0的个数
    even_indices = list(range(0, len(step_scores), 2))
    count_ones = sum(step_scores)
    count_zeros = len(even_indices) - count_ones

    # 再更新偶数项的值
    for i in even_indices:
        if step_scores[i] == 1:
            if i<len(new_parts)-1:
                if new_parts[i+1] in new_parts[i]:
                    step_scores[i] = 0
                else:
                    step_scores[i] = 0.025
            else:
                step_scores[i] = 0.025
        else:
            step_scores[i] = 0
            
    # 再更新偶数项的值
    if answer is None:
        answer_score = 0.0
    else:
        if em_check(answer, ground_truth['target']):
            answer_score = score
        else:
            answer_score = 0.0
            step_scores[even_indices[-1]] += 0.1
            
    # print("even_indices", even_indices)
    # print("count_ones", count_ones)
    # print("count_zeros", count_zeros)
    # print("step_scores", step_scores)
    # print("answer_score", answer_score)

    return {
        'answer_score': answer_score,
        'step_scores': step_scores,
    }
 




def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth['target']}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
