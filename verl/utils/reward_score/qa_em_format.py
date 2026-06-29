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

# 文本归一化
def normalize_answer(s):
    # 去掉英语冠词
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    # 统一空格格式
    def white_space_fix(text):
        return " ".join(text.split())
    # 去掉所有标点符号
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    # 全部转为小写
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# 计算EM精确匹配分数
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

# 格式校验函数
def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # 检查标签是否配对
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"

# 提取答案
def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

# 提取信息
def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

# 判断检索到的信息中是否包含任意一个标准答案
def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 检查格式是否合法
    is_valid_format, _ = is_valid_sequence(solution_str)
    # 如果格式合法检查检索是否正确
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    # 提取最终答案
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    # 随机打印调试信息（1/64 概率）
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    # 没有答案，格式正确检索正确0.3 格式正确检索错误0.2
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        # 答案正确格式正确1，答案正确格式不正确0.8
        if em_check(answer, ground_truth['target']):
            if is_valid_format:
                return score # 1
            else:
                return score - structure_format_score # 0.8
        # 答案不正确格式正确检索正确0.3，答案不正确格式正确检索不正确0.2
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # 0.3
            else:
                return structure_format_score # 0.2
        else:
            # 答案不正确格式也不正确
            return final_format_score # 0.1
