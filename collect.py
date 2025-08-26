import os
import glob                         # 原脚本把 glob 写成 globs
import torch
import numpy as np
from pathlib import Path

# see `train.sh`
SYSTEMUSER = os.getenv("USER") or "unknown"
expr_name = "AReaL-GRPO-n4-sliding"
expr_dir = f"/tmp/areal/experiments/checkpoints/{SYSTEMUSER}/{expr_name}"

trail_names = os.listdir(expr_dir)
save_root = "./patterns"

for trail_name in trail_names:
    ckpt_dir = Path(expr_dir) / trail_name / "default"

    if not ckpt_dir.exists():
        print(f"Skip {trail_name}: {ckpt_dir} does not exist")
        continue

    # 过滤并排序 checkpoint
    epoch_dirs = [p for p in ckpt_dir.glob("epoch*") if p.is_dir()]
    if not epoch_dirs:
        print(f"Skip {trail_name}: no epoch* dir found")
        continue

    # 按 global_step 升序排序
    epoch_dirs.sort(key=lambda p: int(p.name.split("globalstep")[-1]))

    latest_ckpt_dir = epoch_dirs[-1]

    # 加载权重
    data = torch.load(os.path.join(latest_ckpt_dir, "adapter_weight.pt"), map_location="cpu")

    adapter_weight_dict = data.get("adapter_weight_dict")
    model_name = data.get("model_name")
    if adapter_weight_dict is None:
        print(f"Skip {trail_name}: no adapter_weight_dict")
        continue

    layers = len(adapter_weight_dict)
    adapter_weight_list = []
    for i in range(layers):
        key = f"model.layers.{i}.self_attn.adapter"
        if key not in adapter_weight_dict:
            raise KeyError(f"{key} missing in {latest_ckpt_dir}")
        adapter_weight_list.append(adapter_weight_dict[key])

    adapter_weight = torch.stack(adapter_weight_list).cpu().float().numpy()

    # 创建输出目录
    save_path = Path(save_root) / expr_name / trail_name / "adapter_weights.tsv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(save_path, adapter_weight, delimiter="\t")
    print(f"Saved {save_path}")