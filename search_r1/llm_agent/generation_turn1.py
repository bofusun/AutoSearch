import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self._dbg_trajectory = []          # 每个样本的字符串轨迹
        self._dbg_turn_text = defaultdict(list)  # sample_idx -> [ (turn, text), ... ]

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        # Step 1: 预先编码 "</information>"
        end_tokens = self.tokenizer("</information>\n\n", add_special_tokens=False)['input_ids']
        # Step 2: 不填充，预先next_obs
        next_obs_ids = self.tokenizer(
            next_obs,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )['input_ids']
        # Step 3: 对超出长度的操作
        processed = []
        for ids in next_obs_ids:
            if len(ids) > self.config.max_obs_length:
                truncated = ids[:self.config.max_obs_length - len(end_tokens)] + end_tokens
            else:
                truncated = ids
            # CWY
            # if len(ids) > self.config.max_obs_length:
            #     keep = max(self.config.max_obs_length - len(end_tokens), 0)
            #     truncated = ids[:keep] + end_tokens
            # else:
            #     truncated = ids
            
            processed.append(truncated)
        # Step 4: padding 成相同长度的 tensor
        next_obs_ids = self.tokenizer.pad(
            {'input_ids': processed},
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, #左边的已有内容（比如对话的前半部分）
                prompt_with_mask: torch.Tensor, #和 prompt 等长，但是其中本来是 info 的位置会被 pad_id 替掉的版本
                response: torch.Tensor, #模型新生成的内容
                info: torch.Tensor = None, #追加的一段“信息块”
                pad_to_left: bool = True # 控制 padding 是往左对齐还是往右对齐。默认 True 表示要做“左 pad”（即把真正有内容的 token 推到右边）
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        # 准备两份“待拼接”列表:tensors 最终会拼出真实序列、tensors_with_mask 用来生成“info 被 mask 掉”的版本
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        # 如果提供了 info，把它加进去，并构造一个与 info 同 shape、全为 pad_id 的 info_mask
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        # 真正拼接
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        # 生成“排序掩码”：pad_to_left=True 时，想把所有 pad 移到最左边 → 需要把“非 pad”放右边，因此 mask 为 != pad_id
        # pad_to_left=False 时，想把所有 pad 移到最右边 → 需要把“是 pad”放右边，因此 mask 为 == pad_id
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        # 利用 argsort 得到“把 pad 排到目标侧”的索引
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices) #正常拼接并重新排序后的结果
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices) #把 info 区段全部变成 pad_id 后的拼接结果

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        # import pdb
        # pdb.set_trace()
        batch_size = gen_batch.batch['input_ids'].shape[0]

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # 初始化turn起始位置数组，用-1表示未激活的turn
        turn_starts_position = torch.full((batch_size, self.config.max_turns + 1), -1, dtype=torch.long)
        # 获取初始左半部分的长度（需要加到相对位置上得到绝对位置）
        left_side_length = original_left_side['input_ids'].shape[1]  # max_start_length
        # 已经在右边累计了多少 token
        current_lengths = torch.zeros(batch_size, dtype=torch.long)
        # 这一轮模型response_lengths
        response_lengths = torch.full((batch_size, self.config.max_turns + 1), -1, dtype=torch.long)  
        # 这一轮模型info_lengths
        info_lengths = torch.full((batch_size, self.config.max_turns + 1), -1, dtype=torch.long)

        response_end_positions = torch.full((batch_size, self.config.max_turns + 1), -1, dtype=torch.long)
        turn_end_positions = torch.full((batch_size, self.config.max_turns + 1), -1, dtype=torch.long)
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            active_indices = torch.where(active_mask)[0]
            next_obs_ids = self._process_next_obs(next_obs)
            # 记录每个活跃样本的turn信息
            for i, idx in enumerate(active_indices):
                # 计算实际生成长度（排除pad tokens）
                gen_length = (responses_ids[idx] != self.tokenizer.pad_token_id).sum().item()
                info_length = (next_obs_ids[idx] != self.tokenizer.pad_token_id).sum().item()

                turn_starts_position[idx, step] = left_side_length + current_lengths[idx].item()
                response_lengths[idx, step] = gen_length
                info_lengths[idx, step] = info_length
                start_response = turn_starts_position[idx, step].item()
                # ✅ 这里把“response 最后一个 token 的绝对位置”也记下来
                if gen_length > 0:
                    response_end_positions[idx, step] = start_response + gen_length - 1
                    turn_end_positions[idx, step] = start_response + gen_length + info_length - 1
                current_lengths[idx] += (gen_length + info_length)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            active_indices = torch.where(active_mask)[0]  # 未结束的
            # 记录每个活跃样本的turn信息
            for i, idx in enumerate(active_indices):
                # 计算实际生成长度（排除pad tokens）
                gen_length = (responses_ids[idx] != self.tokenizer.pad_token_id).sum().item()
                
                turn_starts_position[idx, self.config.max_turns] = left_side_length + current_lengths[idx].item()
                response_lengths[idx, self.config.max_turns] = gen_length
                start_response = turn_starts_position[idx, self.config.max_turns].item()
                # ✅ 最后一次的 response 结束位置
                if gen_length > 0:
                    response_end_positions[idx, self.config.max_turns] = start_response + gen_length - 1
                    turn_end_positions[idx, self.config.max_turns] = start_response + gen_length - 1                    
                current_lengths[idx] += gen_length

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)


            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['turn_starts_position'] = turn_starts_position.tolist()
        # print("[LLM_LOOP] turn_starts_position =", meta_info['turn_starts_position'])
        meta_info['response_lengths'] = response_lengths.tolist()
        meta_info['info_lengths'] = info_lengths.tolist()

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # # 把右半边拿出来，形状是 [B, T_right]
        # right_concat = original_right_side['responses']  # torch.LongTensor
        # B = right_concat.size(0)
        # left_len = left_side_length
        # max_turns = self.config.max_turns

        # for b in range(B):
        #     print(f"\n==== Sample {b} ====")
        #     for t in range(max_turns + 1):   # +1 是你最后那次 final rollout 也记录了
        #         start_abs = turn_starts_position[b, t].item()
        #         if start_abs == -1:
        #             continue  # 这个 turn 没激活/没生成
        #         resp_len = response_lengths[b, t].item()
        #         info_len = info_lengths[b, t].item()

        #         # 把“绝对位置”转成“右半边位置”
        #         start_right = start_abs - left_len
        #         if start_right < 0:
        #             # 理论上不会，但防御一下
        #             print(f"  [turn {t}] start_right < 0, skip")
        #             continue

        #         # 取出这一 turn 的 response token
        #         resp_tokens = right_concat[b, start_right : start_right + resp_len]

        #         # 取出这一 turn 的 info token（如果有）
        #         info_tokens = None
        #         if info_len > 0:
        #             info_tokens = right_concat[b, start_right + resp_len : start_right + resp_len + info_len]

        #         # decode
        #         resp_text = self.tokenizer.decode(
        #             resp_tokens.tolist(),
        #             skip_special_tokens=False   # 你这里最好先别跳，方便看标签
        #         )
        #         print(f"  [turn {t}] response ({resp_len}): {repr(resp_text)}")

        #         if info_tokens is not None:
        #             info_text = self.tokenizer.decode(
        #                 info_tokens.tolist(),
        #                 skip_special_tokens=False
        #             )
        #             print(f"  [turn {t}] info     ({info_len}): {repr(info_text)}")


        return self._compose_final_output(original_left_side, original_right_side, meta_info,turn_starts_position=turn_starts_position,response_end_positions=response_end_positions,turn_end_positions=turn_end_positions)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            turn_starts_position: torch.Tensor,
                            response_end_positions: torch.Tensor,
                            turn_end_positions: torch.Tensor) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        # 注意它们现在是 [B, max_turns+1]，本来就是 tensor，可以直接塞
        final_output['turn_starts_position'] = turn_starts_position
        final_output['turn_ends_position'] = turn_end_positions 
        # 
        final_output['responses_turn'] = response_end_positions 

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference