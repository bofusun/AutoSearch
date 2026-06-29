data_name=nq_hotpotqa_train

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/data/${data_name} # first download the data from https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

WAND_PROJECT="Search-R1"

export BASE_MODEL='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/model/Qwen2.5-3B'
# export BASE_MODEL='/mnt/parallel_ssd/home/zdhs0009/sjb/Search-R1/model/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo'
export EXPERIMENT_NAME=${data_name}-search-r1-ppo-qwen2.5-3b-em-structureformat_4mturn_tuneturn_new78


# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/Search-R1-new

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo_format_turn_new78 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=6500 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=800 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty="low_var_kl" \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['mlflow'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/Search-R1-new/verl_checkpoints/$EXPERIMENT_NAME \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/Search-R1-new/experiments/experiment_log/$EXPERIMENT_NAME.log
