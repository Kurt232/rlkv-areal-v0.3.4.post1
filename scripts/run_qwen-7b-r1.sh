set -e
export CUDA_VISIBLE_DEVICES=0,1

expr_name=AReaL-GRPO-n4-streaming-llama
model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

lr=1e-2
epochs=2
reg_loss_scale=1e-3
reg_loss_tau=0.55
adapter_init_value=1.0
sink_win_size=128
recent_win_size=256
trial_name="qwen7b_cfg0_t0.55__lr1e-2_ep2_bs32_reg1e-3"

python3 -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/rlkv/deepseek_grpo.yaml \
    experiment_name="${expr_name}" \
    trial_name="${trial_name}" \
    total_train_epochs="${epochs}" \
    ++train_dataset.path=Kurt232/Sampled-Laser-Dataset-Qwen-2.5-7B-R1 \
    ++valid_dataset.path=Kurt232/Sampled-Laser-Dataset-Qwen-2.5-7B-R1 \
    ++actor.path="${model}" \
    ++actor.optimizer.lr="${lr}" \
    ++actor.enable_mixed_attn_training="true" \
    ++actor.sink_window_size="${sink_win_size}" \
    ++actor.recent_window_size="${recent_win_size}" \
    ++actor.reg_loss_scale="${reg_loss_scale}" \
    ++actor.reg_loss_tau="${reg_loss_tau}" \
    ++actor.adapter_init_value="${adapter_init_value}" \
    ++sglang.attention_backend="triton" \
    ++sglang.enable_mixed_attention="true" \
    ++sglang.sink_window_size="${sink_win_size}" \
    ++sglang.recent_window_size="${recent_win_size}" \
    ++sglang.adapter_init_value="${adapter_init_value}" \
    ++stats_logger.wandb.mode="online"
