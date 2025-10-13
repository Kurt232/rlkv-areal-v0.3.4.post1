import torch

from realhf.base import stats_tracker


def reg_loss_fn(x: torch.Tensor, reward_scores: torch.Tensor, tau: float) -> torch.Tensor:
    # reward_scores: (batch_size * group_size,)
    # L1 regularization loss
    numel = x.numel()
    l1 = x.abs().sum()

    reward_mean = reward_scores.mean().detach()
    scaling_factor = (torch.exp(reward_mean) - 1) if reward_mean.item() > tau else 0.0
    loss = l1 / numel * scaling_factor
    
    stats_tracker.scalar(
        reg_loss=loss.detach(),
        reg_scaling_factor=scaling_factor,
        reg_reward_mean=reward_mean,
        adapter_weight_mean=x.mean().detach(),
        adapter_weight_std=x.std().detach(),
        adapter_weight_min=x.min().detach(),
        adapter_weight_max=x.max().detach(),
    )

    return loss
