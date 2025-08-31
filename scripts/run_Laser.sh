set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

expr_name=AReaL-GRPO-n4-streaming
model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
trial_name=$1
lr=$2
epochs=1
reg_loss_scale=$3
adapter_init_value=$4
sink_win_size=128
recent_win_size=256
trial_name="laser__${trial_name}"

python3 -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml \
    experiment_name="${expr_name}" \
    trial_name="${trial_name}" \
    total_train_epochs="${epochs}" \
    ++train_dataset.path=Kurt232/Sampled-Laser-Dataset \
    ++valid_dataset.path=Kurt232/Sampled-Laser-Dataset \
    ++actor.path="${model}" \
    ++actor.optimizer.lr="${lr}" \
    ++actor.enable_mixed_attn_training="true" \
    ++actor.sink_window_size="${sink_win_size}" \
    ++actor.recent_window_size="${recent_win_size}" \
    ++actor.reg_loss_scale="${reg_loss_scale}" \
    ++actor.adapter_init_value="${adapter_init_value}" \
    ++sglang.attention_backend="triton" \
    ++sglang.enable_mixed_attention="true" \
    ++sglang.sink_window_size="${sink_win_size}" \
    ++sglang.recent_window_size="${recent_win_size}" \
    ++sglang.adapter_init_value="${adapter_init_value}" \
    ++stats_logger.wandb.mode="online"
