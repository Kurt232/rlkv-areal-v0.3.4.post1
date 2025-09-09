import torch

from realhf.base import stats_tracker


def reg_loss_fn(x: torch.Tensor, reward_scores: torch.Tensor) -> torch.Tensor:
    # reward_scores: (batch_size * group_size,)
    # L1 regularization loss
    numel = x.numel()
    l1 = x.abs().sum()

    reward_mean = reward_scores.mean()
    scaling_factor = (torch.exp(reward_mean) - 1).detach() # assume reward \in [0, 1]
    loss = l1 / numel * scaling_factor
    
    stats_tracker.scalar(
        reg_loss=loss.detach(),
        reg_scaling_factor=scaling_factor.detach(),
        reg_reward_mean=reward_mean.detach(),
        adapter_weight_mean=x.mean().detach(),
        adapter_weight_std=x.std().detach(),
        adapter_weight_min=x.min().detach(),
        adapter_weight_max=x.max().detach(),
    )

    return loss
