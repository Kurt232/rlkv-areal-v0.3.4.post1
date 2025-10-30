import os

import torch


def clamp_adapter_weight(model, v_min=0, v_max=1):
    for layer in model.model.layers:
        module = layer.self_attn
        if not hasattr(module, "adapter") or module.adapter is None:
            continue
        module.adapter.clamp(v_min, v_max)


def get_adapter_weight(model):
    adapter_weight = []

    for layer in model.model.layers:
        module = layer.self_attn
        if not hasattr(module, "adapter") or module.adapter is None:
            continue
        adapter_weight.append(module.adapter.weight)

    if len(adapter_weight) != len(model.model.layers):
        raise ValueError(f"Not all layers have adapter weights. ")

    return adapter_weight


def save_adapter_weight(model_name, state_dict, save_dir):
    adapter_weight_dict = {}
    for key, value in state_dict.items():
        if "adapter" in key:
            adapter_weight_dict[key] = value
    if len(adapter_weight_dict) == 0:
        raise ValueError("No adapter weight found")

    torch.save(
        {
            "adapter_weight_dict": adapter_weight_dict,
            "model_name": model_name,
        },
        os.path.join(save_dir, "adapter_weight.pt"),
    )


def load_adapter_weight(path):
    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt.get("model_name", None)
    adapter_weight_dict = ckpt.get("adapter_weight_dict", None)
    if model_name is None or adapter_weight_dict is None:
        raise ValueError("Invalid checkpoint format")
    return model_name, adapter_weight_dict
