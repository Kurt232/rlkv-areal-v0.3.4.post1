from .forward import (
    enable_llama_mixed_attention_training,
    enable_qwen2_mixed_attention_training,
    enable_qwen3_mixed_attention_training,
)
from .utils import (
    clamp_adapter_weight,
    get_adapter_weight,
    load_adapter_weight,
    save_adapter_weight,
)


def enable_mixed_attention_training(
    model,
    sink_window_size: int = 128,
    recent_window_size: int = 256,
    adapter_init_value: float = 1.0,
    ulysses_sp_size: int = 1,
):
    if "llama" in model.config.model_type:
        enable_llama_mixed_attention_training(
            model,
            sink_window_size,
            recent_window_size,
            adapter_init_value,
            ulysses_sp_size,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen2_mixed_attention_training(
            model,
            sink_window_size,
            recent_window_size,
            adapter_init_value,
            ulysses_sp_size,
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_mixed_attention_training(
            model,
            sink_window_size,
            recent_window_size,
            adapter_init_value,
            ulysses_sp_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


__all__ = [
    "clamp_adapter_weight",
    "get_adapter_weight",
    "load_adapter_weight",
    "save_adapter_weight",
    "enable_mixed_attention_training",
]
