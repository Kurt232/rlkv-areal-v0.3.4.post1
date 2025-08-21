set -e
export CUDA_VISIBLE_DEVICES=0,1

model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
expr_name="AReaL-GRPO"
trial_name="test"
sink_win_size=128
recent_win_size=256
reg_loss_scale=1
lr=0.02
epochs=100

python3 -m areal.launcher.local examples/lite/gsm8k_grpo.py --config examples/lite/configs/gsm8k_grpo.yaml \
    experiment_name=${expr_name} \
    trial_name=${trial_name} \
    allocation_mode=sglang.d1p1t1+d1p1t1 \
    total_train_epochs=${epochs} \
    ++actor.path=${model} \
    ++actor.optimizer.lr=${lr} \
    ++actor.enable_mixed_attn_training=true \
    ++actor.sink_window_size=${sink_win_size} \
    ++actor.recent_window_size=${recent_win_size} \
    ++actor.reg_loss_scale=${reg_loss_scale} \
    ++sglang.attention_backend=triton \
    ++sglang.enable_mixed_attention=true \
    ++sglang.sink_window_size=${sink_win_size} \
    ++sglang.recent_window_size=${recent_win_size}
