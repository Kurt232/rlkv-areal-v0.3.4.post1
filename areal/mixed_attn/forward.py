from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
from block_sparse_attn import block_streaming_attn_func
from flash_attn import flash_attn_varlen_func
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
)
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.models.phi3.modeling_phi3 import (
    apply_rotary_pos_emb as phi3_apply_rotary_pos_emb,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from realhf.base import logging

logger = logging.getLogger("Mixed Attention")


def enable_mixed_attention_training(
    model,
    forward_fn,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
):
    # areal don't support TP
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    num_sink_blocks = (sink_window_size + 127) // 128
    num_recent_blocks = (recent_window_size + 127) // 128
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads

    logger.info(
        f"Using blocksparse implementation with {num_sink_blocks} sink blocks, {num_recent_blocks} recent blocks, and {num_heads} heads per device"
    )
    streaming_mask = torch.tensor(
        [num_sink_blocks, num_recent_blocks] * num_heads,
        device=device,
        dtype=torch.int32,
    )

    for layer in model.model.layers:
        module = layer.self_attn
        module.forward = types.MethodType(forward_fn, module)
        if "adapter" not in module._parameters:
            # Original
            module.register_parameter(
                "adapter",
                nn.Parameter(
                    torch.ones(
                        num_kv_heads,
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    * adapter_init_value
                ),
            )
        # else: means the adapter is already registered, e.g. middle checkpoint of training

        module.register_buffer("streaming_mask", streaming_mask)
        module.register_buffer(
            "head_mask_type",
            torch.tensor([-1] * num_heads, device=device, dtype=torch.int32),
        )


def llama_mixed_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],  # add from AReaL
    cu_seqlens_k: Optional[torch.Tensor],  # add from AReaL
    cu_seqlens: Optional[
        torch.Tensor
    ],  # add from AReaL, eq to cu_seqlens_q and cu_seqlens_k
    max_seqlen: Optional[int] = None,  # add from AReaL
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    total_tokens = input_shape[0] * input_shape[1]
    query_states = query_states.transpose(1, 2).reshape(total_tokens, -1, self.head_dim)
    key_states = key_states.transpose(1, 2).reshape(total_tokens, -1, self.head_dim)
    value_states = value_states.transpose(1, 2).reshape(total_tokens, -1, self.head_dim)

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        softmax_scale=self.scaling,
        causal=True,
    )

    streaming_attn_output = block_streaming_attn_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k_=max_seqlen,
        max_seqlen_q_=max_seqlen,
        p_dropout=0.0 if not self.training else self.attention_dropout,
        streaming_info=self.streaming_mask,
        head_mask_type=self.head_mask_type,
        softmax_scale=self.scaling,
        is_causal=True,
    )

    # adapter shape: [num_heads] -> [1, num_heads, 1]
    adapter = self.adapter.repeat_interleave(self.num_key_value_groups)
    adapter = adapter.view(1, -1, 1)

    attn_output = adapter * attn_output + (1.0 - adapter) * streaming_attn_output
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def enable_llama_mixed_attention_training(
    model: LlamaForCausalLM,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
):
    enable_mixed_attention_training(
        model,
        llama_mixed_attention_forward,
        sink_window_size,
        recent_window_size,
        adapter_init_value,
    )


def phi3_mixed_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],  # add from AReaL
    cu_seqlens_k: Optional[torch.Tensor],  # add from AReaL
    cu_seqlens: Optional[
        torch.Tensor
    ],  # add from AReaL, eq to cu_seqlens_q and cu_seqlens_k
    max_seqlen: Optional[int] = None,  # add from AReaL
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.config.num_attention_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[
        ..., query_pos : query_pos + self.num_key_value_heads * self.head_dim
    ]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = phi3_apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    total_tokens = input_shape[0] * input_shape[1]
    query_states = query_states.transpose(1, 2).reshape(total_tokens, -1, self.head_dim)
    key_states = key_states.transpose(1, 2).reshape(total_tokens, -1, self.head_dim)
    value_states = value_states.transpose(1, 2).reshape(total_tokens, -1, self.head_dim)

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        softmax_scale=self.scaling,
        causal=True,
    )

    streaming_attn_output = block_streaming_attn_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k_=max_seqlen,
        max_seqlen_q_=max_seqlen,
        p_dropout=0.0 if not self.training else self.attention_dropout,
        streaming_info=self.streaming_mask,
        head_mask_type=self.head_mask_type,
        softmax_scale=self.scaling,
        is_causal=True,
    )

    # adapter shape: [num_heads] -> [1, num_heads, 1]
    adapter = self.adapter.repeat_interleave(self.num_key_value_groups)
    adapter = adapter.view(1, -1, 1)

    attn_output = adapter * attn_output + (1.0 - adapter) * streaming_attn_output
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def enable_phi3_mixed_attention_training(
    model: Phi3ForCausalLM,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
):
    enable_mixed_attention_training(
        model,
        phi3_mixed_attention_forward,
        sink_window_size,
        recent_window_size,
        adapter_init_value,
    )


def enable_qwen2_mixed_attention_training(
    model: Qwen2ForCausalLM,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
):
    enable_mixed_attention_training(
        model,
        llama_mixed_attention_forward,  # Qwen2 uses the same forward as Llama
        sink_window_size,
        recent_window_size,
        adapter_init_value,
    )
