import torch

from realhf.base import stats_tracker


def reg_loss_fn(x: torch.Tensor):
    # L1 regularization loss
    numel = x.numel()
    l1 = x.abs().sum()
    loss = l1 / numel

    stats_tracker.scalar(
        reg_loss=loss.detach(),
        adapter_weight_mean=x.mean().detach(),
        adapter_weight_std=x.std().detach(),
        adapter_weight_min=x.min().detach(),
        adapter_weight_max=x.max().detach(),
    )

    return loss
