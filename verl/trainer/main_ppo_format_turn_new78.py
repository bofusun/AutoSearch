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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
import re
import string
from typing import Union, List, Tuple

import torch
import numpy as np
from collections import Counter
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem, collate_fn
from verl.utils.reward_score import qa_em_format_turn62, qa_em_turn19
from verl.utils.reward_score import qa_em, qa_f1
from verl.trainer.ppo.ray_trainer_turn64 import RayPPOTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def _select_rm_score_fn(data_source):
    return qa_em_format_turn62.compute_score_em

def _select_rm_score_em1(data_source):
    return qa_em_turn19.em_check

def _select_rm_score_em(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError

def _select_rm_score_f1(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_f1.compute_score_f1
    else:
        raise NotImplementedError
    
def _extract_qwen_solution(sequences_str):
    # Extract the solution from the sequences string
    
    # The solution is the string after the first occurrence of "Assistant"
    return sequences_str.split("<|im_start|>assistant")[-1].strip() 

def _prepare_qa_generation_batch(
    prompts_text: List[str],
    tokenizer,
    size_divisor: int = None,
) -> DataProto:

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_size = len(prompts_text)

    # pad the batch to the size_divisor
    if size_divisor and batch_size % size_divisor != 0:
        need_to_pad = size_divisor - batch_size % size_divisor
        prompts_text = prompts_text + [prompts_text[0]] * need_to_pad
        batch_size = len(prompts_text)

    encoded = tokenizer(
        prompts_text,
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 构造 position_ids
    position_ids = torch.zeros_like(input_ids)
    for i in range(batch_size):
        non_pad_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            first_token_pos = non_pad_indices[0]
            seq_len = attention_mask[i].sum()
            position_ids[i, first_token_pos:first_token_pos + seq_len] = torch.arange(seq_len)

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "recompute_log_prob": False,
    }

    tokenizer.padding_side = original_padding_side

    return DataProto(batch=batch, meta_info=meta_info)


def _decode_qa_generation_output(gen_output: DataProto, tokenizer) -> Tuple[List[str], List[str]]:

    inputs, outputs = [], []

    if isinstance(gen_output, DataProtoItem):
        batch_td = gen_output.batch
        batch_size = batch_td.batch_size[0]

        for i in range(batch_size):
            item_td = batch_td[i]

            prompt_ids = item_td["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = item_td["attention_mask"]

            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item_td["responses"]
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            inputs.append(tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
            outputs.append(tokenizer.decode(valid_response_ids, skip_special_tokens=False))
    else:
        batch_size = len(gen_output)

        for i in range(batch_size):
            data_item = gen_output[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            attention_mask = data_item.batch["attention_mask"]

            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            inputs.append(tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
            outputs.append(tokenizer.decode(valid_response_ids, skip_special_tokens=False))

    return inputs, outputs


def ensure_batch_is_padded(padded_batch, original_batch, divisor, original_pad_size):

    if len(padded_batch) % divisor != 0:
        print(
            f"⚠️ Original padding failed! (Size: {len(padded_batch)}, Divisor: {divisor}). "
            f"Applying a robust fix..."
        )
        original_size = len(original_batch)
        if original_size == 0:
            return original_batch, 0

        new_pad_size = divisor - (original_size % divisor)

        padding_items = [original_batch[i % original_size] for i in range(new_pad_size)]
        padding_data = collate_fn(padding_items)

        fixed_padded_batch = DataProto.concat([original_batch, padding_data])

        print(f"✅ Fix applied. Batch resized from {original_size} to {len(fixed_padded_batch)}.")

        return fixed_padded_batch, new_pad_size

    return padded_batch, original_pad_size


def generate_extra_outputs_for_qa_em(
    input_strings: Union[str, List[str]],
    actor_rollout_wg,
    tokenizer,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    batch_size: int = 128,
    **kwargs,
) -> Tuple[List[str], List[str]]:

    if isinstance(input_strings, str):
        input_list = [input_strings]
    else:
        input_list = input_strings

    if not input_list:
        return [], []

    all_inputs, all_outputs = [], []

    for i in range(0, len(input_list), batch_size):
        batch_inputs = input_list[i: i + batch_size]

        gen_batch = _prepare_qa_generation_batch(
            batch_inputs, tokenizer, getattr(actor_rollout_wg, "world_size", None)
        )

        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            **kwargs,
        }

        if hasattr(actor_rollout_wg, "world_size"):
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, actor_rollout_wg.world_size
            )
        else:
            gen_batch_padded, pad_size = gen_batch, 0

        gen_batch_padded, pad_size = ensure_batch_is_padded(
            gen_batch_padded, gen_batch, actor_rollout_wg.world_size, pad_size
        )

        gen_output_padded = actor_rollout_wg.generate_sequences(gen_batch_padded)
        gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size) if pad_size else gen_output_padded

        batch_inputs_decoded, batch_outputs = _decode_qa_generation_output(gen_output, tokenizer)
        all_inputs.extend(batch_inputs_decoded)
        all_outputs.extend(batch_outputs)

    return all_inputs, all_outputs

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

class RewardManager():
    """The reward manager.
    """
    # 从回答中提取答案
    def _extract_final_answer(self, text: str) -> str:
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    # 从轨迹中提取问题
    def _extract_original_question(self, prompt_text: str) -> str:
        # 先全局找 Question: 后面的内容，直到换行、<|im_end|> 或文本结束
        m = re.search(
            r"Question:\s*(.*?)(?:\n|<\|im_end\|>|<\|im_start\|assistant|$)",
            prompt_text,
            flags=re.DOTALL,
        )
        question_body = m.group(1).strip()
        return question_body
        
    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., process_reward_ratio=0., is_validation: bool = False):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.process_reward_ratio = process_reward_ratio
        self.is_validation = is_validation

    def __call__(self, data: DataProto, actor_rollout_wg=None):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        reward_tensor_list = []
        already_print_data_sources = {}
        
        # -------- 收集“每轮上下文”的全局容器（用于一次性送给 actor_rollout_wg） --------
        all_turn_contexts: List[str] = []               # 所有样本所有轮的上下文文本，用于批量生成
        ctx_index_mapping: List[Tuple[int, int]] = []   # (样本 i, 该样本的第 local_turn 个 turn)
        turn_context_texts = {}                         # (i, local_t) -> answer_prompt 文本

        # -------- 为后续 early-stop / 统计准备的 per-sample 缓存 --------
        sample_info_end_positions_right = []            # 每个样本：info 结束在 responses 里的下标列表
        sample_reward_end_positions_right = []          # 每个样本：对应 info 的 reward 位置（answer end）
        sample_reward_positions_right = []              # 每个样本：每轮 answer end 的位置（所有 turn）
        sample_per_turn_responses = []                  # 每个样本：每轮 answer 文本
        sample_per_turn_infos = []                      # 每个样本：每轮 info 文本
        sample_sequences_str = []                       # 每个样本：完整 decode 的 prompt+response
        sample_valid_response_length = []               # 每个样本：有效 response 长度
        sample_answer_scores = []                       # 每个样本：RM 的最终 answer_score
        sample_step_scores = []                       # 每个样本：RM 的最终 answer_score
        sample_data_sources = []                        # 每个样本：data_source
        sample_reward_tensors = []                      # 每个样本：只包含 step / novelty 奖励的 reward_tensor

        # ground_truth 在 early-stop 会用到，提前做一次 normalize
        sample_normalized_gt = []                       # 每个样本：归一化后的 ground_truth target 字符串
        
        reward_em = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_f1 = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        early_corrects = torch.zeros([data.batch['responses'].shape[0]], dtype=torch.float32)
        early_correct_ratios = torch.zeros([data.batch['responses'].shape[0]], dtype=torch.float32)

        # -------- 第一次大循环：解析 turn + 构造上下文 + 调 RM 打分（不含 early-stop） --------
        for i in range(len(data)):
            data_item = data[i]

            # 1. 提取基本数据
            prompt_ids = data_item.batch['prompts']          # [L_left]
            responses_all = data_item.batch['responses']     # [L_right]
            left_len = prompt_ids.shape[-1]

            turn_starts = data.batch['turn_starts_position'][i]   # [T]
            turn_ends   = data.batch['turn_ends_position'][i]     # [T]
            resp_ends   = data.batch['responses_turn'][i]         # [T]

            # 2. per-turn 临时容器
            reward_positions_right: List[int] = []
            info_positions_right: List[int] = []
            reward_positions_for_info: List[int] = []
            per_turn_responses: List[str] = []
            per_turn_infos: List[str] = []

            # 3. decode prompt，提取原始问题
            valid_prompt_length = data_item.batch['attention_mask'][:left_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            full_prompt_text = self.tokenizer.decode(
                valid_prompt_ids,
                skip_special_tokens=False
            )
            original_question_text = self._extract_original_question(full_prompt_text)

            # 4. responses 侧：有效长度 / 序列字符串
            response_ids = responses_all
            valid_response_length = data_item.batch['attention_mask'][left_len:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids), dim=0)
            sequences_str = self.tokenizer.decode(sequences)
            sample_sequences_str.append(sequences_str)
            sample_valid_response_length.append(valid_response_length.item())

            # 5. ground_truth & data_source
            ground_truth_full = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            sample_data_sources.append(data_source)
            
            em_fn = _select_rm_score_em(data_source)
            f1_fn = _select_rm_score_f1(data_source)

            em_score = em_fn(solution_str=sequences_str, ground_truth=ground_truth_full, format_score=0)
            f1_score = f1_fn(solution_str=sequences_str, ground_truth=ground_truth_full, format_score=0)

            reward_em[i, valid_response_length - 1] = em_score
            reward_f1[i, valid_response_length - 1] = f1_score

            # 提前处理 EM 用的 gt（target -> str -> normalize）
            gt = ground_truth_full.get("target", "")
            if isinstance(gt, (list, np.ndarray)):
                gt = gt[0] if len(gt) > 0 else ""
            gt = qa_em_turn19.normalize_answer(str(gt))
            sample_normalized_gt.append(gt)

            # 6. 解析每个 turn，构造 per-turn 文本 + info_end_positions，同时构造上下文 prompt
            cumulative_text = ""  # 累积本样本中“之前所有轮的 answer + info”文本

            for t in range(len(resp_ends.tolist())):
                # padding: -1 表示没有真实 turn
                if resp_ends[t] == -1 or turn_starts[t] == -1:
                    continue

                start_right = turn_starts[t] - left_len
                end_right = resp_ends[t] - left_len
                turn_end_right = turn_ends[t] - left_len

                reward_positions_right.append(end_right)

                # 当前轮的 answer token / 文本
                resp_tokens = responses_all[start_right: end_right + 1]
                resp_text = self.tokenizer.decode(
                    resp_tokens.tolist(),
                    skip_special_tokens=False
                )
                per_turn_responses.append(resp_text)
                cumulative_text += resp_text

                # 如果该轮有 <information>
                has_info = (turn_end_right > end_right)
                if has_info:
                    info_tokens = responses_all[end_right + 1: turn_end_right + 1]
                    info_text = self.tokenizer.decode(
                        info_tokens.tolist(),
                        skip_special_tokens=False
                    )
                    per_turn_infos.append(info_text)
                    cumulative_text += info_text

                    info_positions_right.append(turn_end_right)
                    reward_positions_for_info.append(end_right)

                    context_text = (
                        "Question: "+ original_question_text + "\n\n" +
                        "Below are your previous reasoning, search calls, and retrieved informations:\n"
                        + cumulative_text + "\n\n"
                    )
                    answer_prompt = (
                        "<|im_start|>system\n"
                        "You are a helpful assistant."
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        "Answer the given question according to search trajectories, which consists of multiple reasoning, search calls, and retrieved informations.\n"
                        "Important instructions:\n"
                        "1) You must conduct reasoning inside <think> and </think> first.\n"
                        "2) After reasoning, output the final answer wrapped in <answer> and </answer>.\n"
                        "For example: <think>Reasoning</think><answer>Jaden Smith</answer>\n"
                        f"Search trajectory: {context_text}."
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"       
                        )  
                    local_t = len(info_positions_right) - 1
                    all_turn_contexts.append(answer_prompt)
                    ctx_index_mapping.append((i, local_t))
                    turn_context_texts[(i, local_t)] = answer_prompt

            # 记录 per-sample 的 info / reward 位置
            sample_info_end_positions_right.append(info_positions_right)
            sample_reward_end_positions_right.append(reward_positions_for_info)
            sample_reward_positions_right.append(reward_positions_right)
            sample_per_turn_responses.append(per_turn_responses)
            sample_per_turn_infos.append(per_turn_infos)

            # 7. 调 RM 打分（只算 step / novelty + answer_score，不立刻放到 token 上）
            responses_all = data_item.batch['responses']
            reward_tensor = torch.zeros_like(
                responses_all, dtype=torch.float32, device=responses_all.device
            )

            compute_score_fn = _select_rm_score_fn(data_source)


            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth_full,
                new_parts=per_turn_responses,
                per_turn_infos=per_turn_infos,
                support_docs=None,
                structure_format_score=self.structure_format_score,
                final_format_score=self.final_format_score,
                retrieval_score=self.retrieval_score
            )

            # step_scores / novelty_rewards 直接打在 reward_tensor 上
            if 'step_scores' in score:
                reward_tensor += self.process_reward_ratio * self._compute_step_reward_tensor(
                    reward_tensor,
                    score.get('step_scores', []),
                    reward_positions_right
                )
            if 'novelty_rewards' in score:
                reward_tensor += self.process_reward_ratio * self._compute_step_reward_tensor(
                    reward_tensor,
                    score.get('novelty_rewards', []),
                    reward_positions_for_info
                )
            last_token_idx = valid_response_length - 1
            reward_tensor[valid_response_length - 1] += score['answer_score']
            
            # 先不处理 answer_score，留到 early-stop 阶段统一放
            sample_answer_scores.append(score['answer_score'])
            sample_step_scores.append(score['step_scores'])


            # 记录该样本的基础 reward_tensor
            sample_reward_tensors.append(reward_tensor)

            # 统计打印计数
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        # -------- 第二步：用 actor_rollout_wg 对“每轮上下文”生成答案 --------
        per_sample_per_turn_answer = {}  # dict[i][local_t] = answer_text

        if actor_rollout_wg is not None and len(all_turn_contexts) > 0:
            _, turn_answers = generate_extra_outputs_for_qa_em(
                input_strings=all_turn_contexts,
                actor_rollout_wg=actor_rollout_wg,
                tokenizer=self.tokenizer,
                temperature=1.0,
                max_prompt_length=512,
            )

            for (sample_idx, local_turn_idx), ans in zip(ctx_index_mapping, turn_answers):
                per_sample_per_turn_answer.setdefault(sample_idx, {})[local_turn_idx] = ans

                # 之后把下面这段打印关掉或加一个 flag 控制，不然 I/O 也很慢
                gt = sample_normalized_gt[sample_idx]
                ctx_text = turn_context_texts[(sample_idx, local_turn_idx)]
                ans_clean = self._extract_final_answer(ans)

                # print("-" * 80)
                # print(f"[DEBUG] sample {sample_idx}, turn {local_turn_idx}")
                # print("[DEBUG] 送给当前模型的上下文（answer_prompt）:")
                # print(ctx_text)
                # print("[DEBUG] 模型在这一轮生成的答案:")
                # print(ans_clean)
                # print("[DEBUG] 该样本的 ground_truth:")
                # print(gt)
                # print("[DEBUG] EM 匹配结果 (model vs ground_truth):",
                #     qa_em_turn19.em_check(ans_clean, gt))
                # print("-" * 80)

        # -------- 第三步：early-stop + 把 answer_score 搬到正确的 token 上 --------
        total_early_correct = 0
        samples_with_info = 0

        final_reward_tensors = []

        for i in range(len(data)):
            reward_tensor = sample_reward_tensors[i]
            info_end_positions = sample_info_end_positions_right[i]
            reward_end_positions = sample_reward_end_positions_right[i]
            reward_end_positions_turn = sample_reward_positions_right[i]
            valid_response_length = sample_valid_response_length[i]
            data_source = sample_data_sources[i]

            early_correct_turn = None

            if i in per_sample_per_turn_answer and len(info_end_positions) >= 1:
                samples_with_info += 1
                turn_answers_dict = per_sample_per_turn_answer[i]
                gt = sample_normalized_gt[i]

                compute_score_em = _select_rm_score_em1(data_source)

                f1_score_gains = []
                last_answer_f1_score = 0
                for local_t in sorted(turn_answers_dict.keys()):
                    raw_ans = turn_answers_dict[local_t]
                    ans_text = self._extract_final_answer(raw_ans)
                    answer_f1_score = self.query_f1_score(prediction=ans_text, golden_answers=gt)
                    f1_score_gain = max(0, answer_f1_score-last_answer_f1_score)
                    last_answer_f1_score = max(answer_f1_score, last_answer_f1_score)
                    f1_score_gains.append(f1_score_gain)
                    # f1增益奖励
                    pos = reward_end_positions_turn[local_t]
                    if 0 <= pos < valid_response_length and (not self.is_validation):
                        reward_tensor[pos] += f1_score_gain
                    # 保持和原逻辑一致：只考虑倒数第二轮之前的 info
                    if local_t >= len(info_end_positions) - 1:
                        continue
                    # 停止
                    if compute_score_em(ans_text, gt):
                        early_correct_turn = local_t
                        early_corrects[i] = 1.0
                        early_correct_ratios[i] = (len(info_end_positions)-local_t-1) / len(info_end_positions)
                        break

            if i in per_sample_per_turn_answer:
                for t in range(len(reward_end_positions_turn)):
                    pos = reward_end_positions_turn[t]
                    # 如果没答对，鼓励搜索
                    if sample_answer_scores[i] == 0:
                        if 0 <= pos < valid_response_length and sample_step_scores[i][t]>=0 and (not self.is_validation):
                            reward_tensor[pos] += 0.0125
                    # 如果答对了，奖励没有early-stop部分
                    else:
                        # 没搜索，直接答对了奖励0.6
                        if len(reward_end_positions_turn) == 1 and (not self.is_validation):
                            reward_tensor[pos] += 0.2
                        # 搜索答对了，搜索越多，奖励越小
                        else:
                            # 搜索了，earlystop, 则early_correct_turn之前奖励
                            if early_correct_turn is not None:
                                if 0 <= t <= early_correct_turn and sample_step_scores[i][t]>=0 and (not self.is_validation):
                                    reward_tensor[pos] += 0.5*(0.4/(early_correct_turn+1)-0.05)
                                else:
                                    break
                            # 搜索了，没earlystop, 则最终回答之前奖励
                            else:
                                if 0 <= pos < valid_response_length and sample_step_scores[i][t]>=0 and (not self.is_validation):
                                    reward_tensor[pos] += 0.5*(0.4/(len(reward_end_positions_turn)-1)-0.05)
                                   
            if early_correct_turn is not None:
                total_early_correct += 1
                info_pos_right = info_end_positions[early_correct_turn]
                reward_pos_right = reward_end_positions[early_correct_turn]
                        
                for t in range(early_correct_turn + 1, len(reward_end_positions_turn)):
                    pos = reward_end_positions_turn[t]
                    if 0 <= pos < valid_response_length and (not self.is_validation):
                        # 覆盖该轮 step/novelty 的 reward，变成固定惩罚
                        reward_tensor[pos] = -0.05

            final_reward_tensors.append(reward_tensor)

        # -------- 统计信息 + 返回结果 --------
        reward_tensor = torch.stack(final_reward_tensors, dim=0)
        data.batch["reward_em"] = reward_em
        data.batch["reward_f1"] = reward_f1
        data.batch["early_corrects"] = early_corrects
        data.batch["early_correct_ratios"] = early_correct_ratios

        if (not self.is_validation) and samples_with_info > 0:
            print(
                "[RewardManager] early-stop ratio: "
                f"{total_early_correct}/{samples_with_info} "
                f"= {total_early_correct / samples_with_info:.4f}"
            )

        return reward_tensor

    # 把计算得到的step_scores放到各个token位置
    def _compute_step_reward_tensor(self, reward_tensor, step_scores: list[float], reward_positions_right: list[int]):

        reward = torch.zeros_like(reward_tensor)
        for i in range(min(len(step_scores), len(reward_positions_right))):
            idx = reward_positions_right[i]
            if 0 <= idx < len(reward_tensor):
                reward[idx] = step_scores[i]
        return reward
    
    def query_f1_score(self, prediction, golden_answers):
        def _tokenize(text):
            return text.split()

        if prediction is None:
            return 0.0
        prediction = normalize_answer(prediction)
        pred_tokens = prediction.split()
        pred_counter = Counter(pred_tokens)

        if type(golden_answers) == str:
            golden_answers = [golden_answers]
        golden_answers = [normalize_answer(answer) for answer in golden_answers]

        max_f1 = 0.0
        for answer in golden_answers:
            ans_tokens = answer.split()
            ans_counter = Counter(ans_tokens)

            common = pred_counter & ans_counter
            overlap = sum(common.values())

            if overlap == 0:
                continue

            precision = overlap / len(pred_tokens) if pred_tokens else 0.0
            recall = overlap / len(ans_tokens) if ans_tokens else 0.0

            if (precision + recall) == 0:
                current_f1 = 0.0
            else:
                current_f1 = 2 * (precision * recall) / (precision + recall)
            
            max_f1 = max(max_f1, current_f1)

        return round(max_f1, 4) 

    
import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer_tune', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'TOKENIZERS_PARALLELISM': 'true','NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer_turn64 import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, config=config)

    # # Note that we always use function-based RM for validation
    # val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, config=config, process_reward_ratio=0.0)
    
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score,
                              process_reward_ratio=1,is_validation=False)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, is_validation=True)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
