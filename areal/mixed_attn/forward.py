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
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from areal.utils import logging
from areal.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)

logger = logging.getLogger("Mixed Attention")


class HeadAdapterLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert num_heads % num_kv_heads == 0
        self.num_kv_groups = num_heads // num_kv_heads

        self.weight = nn.Parameter(
            torch.ones(num_kv_heads, dtype=params_dtype, requires_grad=True)
        )

    def forward(self, o: torch.Tensor, o_streaming: torch.Tensor) -> torch.Tensor:
        # adapter shape: [num_heads] -> [1, 1, num_heads, 1]
        adapter = self.weight.repeat_interleave(self.num_kv_groups).view(1, 1, -1, 1)
        return adapter * o + (1.0 - adapter) * o_streaming

    @torch.no_grad()
    def clamp(self, v_min=0, v_max=1):
        self.weight.clamp_(v_min, v_max)


def enable_mixed_attention_training(
    model,
    forward_fn,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
    ulysses_sp_size: int = 1,
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
        if "adapter" not in module._modules:
            adapter = HeadAdapterLayer(
                num_heads,
                num_kv_heads,
                params_dtype=dtype,
            )
            with torch.no_grad():
                adapter.weight.fill_(adapter_init_value)
            module.add_module("adapter", adapter)
        # else: means the adapter is already registered, e.g. middle checkpoint of training

        module.register_buffer("streaming_mask", streaming_mask)
        module.register_buffer(
            "head_mask_type",
            torch.tensor([-1] * num_heads, device=device, dtype=torch.int32),
        )
        if ulysses_sp_size <= 1:
            module.mixed_attn_func = _mixed_attention_forward_func
        else:
            module.mixed_attn_func = _ulysses_mixed_attention_forward_func


def llama_mixed_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],  # add from AReaL
    cu_seq_lens_k: Optional[torch.Tensor],  # add from AReaL
    cu_seqlens: Optional[
        torch.Tensor
    ],  # add from AReaL, eq to cu_seq_lens_q and cu_seq_lens_k
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

    attn_output, streaming_attn_output = self.mixed_attn_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        softmax_scale=self.scaling,
        streaming_mask=self.streaming_mask,
        head_mask_type=self.head_mask_type,
        causal=True,
    )

    attn_output = self.adapter(attn_output, streaming_attn_output)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def enable_llama_mixed_attention_training(
    model: LlamaForCausalLM,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
    ulysses_sp_size: int = 1,
):
    enable_mixed_attention_training(
        model,
        llama_mixed_attention_forward,
        sink_window_size,
        recent_window_size,
        adapter_init_value,
        ulysses_sp_size,
    )


def enable_qwen2_mixed_attention_training(
    model: Qwen2ForCausalLM,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
    ulysses_sp_size: int = 1,
):
    enable_mixed_attention_training(
        model,
        llama_mixed_attention_forward,  # Qwen2 uses the same forward as Llama
        sink_window_size,
        recent_window_size,
        adapter_init_value,
        ulysses_sp_size,
    )


def qwen3_mixed_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],  # add from AReaL
    cu_seq_lens_k: Optional[torch.Tensor],  # add from AReaL
    cu_seqlens: Optional[
        torch.Tensor
    ],  # add from AReaL, eq to cu_seq_lens_q and cu_seq_lens_k
    max_seqlen: Optional[int] = None,  # add from AReaL
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_output, streaming_attn_output = self.mixed_attn_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        softmax_scale=self.scaling,
        streaming_mask=self.streaming_mask,
        head_mask_type=self.head_mask_type,
        causal=True,
    )

    attn_output = self.adapter(attn_output, streaming_attn_output)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def enable_qwen3_mixed_attention_training(
    model: Qwen3ForCausalLM,
    sink_window_size: int,
    recent_window_size: int,
    adapter_init_value: float = 1.0,
    ulysses_sp_size: int = 1,
):
    enable_mixed_attention_training(
        model,
        qwen3_mixed_attention_forward,
        sink_window_size,
        recent_window_size,
        adapter_init_value,
        ulysses_sp_size,
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep).
    The hidden states go from (batch, seqlen, num_key_value_heads, head_dim)
    to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_mixed_attention_forward_func(
    query_states,
    key_states,
    value_states,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    streaming_mask,
    head_mask_type,
    causal,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    query = query_states.transpose(1, 2)
    key = key_states.transpose(1, 2)
    value = value_states.transpose(1, 2)

    if ulysses_sp_size > 1:
        repeats = max(ulysses_sp_size // key.size(2), 1)
        key = repeat_kv(key, repeats)
        value = repeat_kv(value, repeats)

        # (1, total_seqlen / sp_size, num_heads, head_dim)
        # -> (1, total_seqlen, num_heads / sp_size, head_dim)
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2)

    query = query.reshape(-1, query.size(-2), query.size(-1))
    key = key.reshape(-1, key.size(-2), key.size(-1))
    value = value.reshape(-1, value.size(-2), value.size(-1))

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )  # [*, n_head/n, head_dim]
    attn_output = attn_output.view(
        query_states.shape[0], -1, attn_output.size(-2), attn_output.size(-1)
    )

    n_heads = query.size(-2)
    streaming_attn_output = block_streaming_attn_func(
        query,
        key,
        value,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k_=max_seqlen_q,
        max_seqlen_q_=max_seqlen_k,
        p_dropout=dropout_p,
        streaming_info=streaming_mask[: 2 * n_heads],
        head_mask_type=head_mask_type[:n_heads],
        softmax_scale=softmax_scale,
        is_causal=causal,
    )  # [*, n_head/n, head_dim]
    streaming_attn_output = streaming_attn_output.view(
        query_states.shape[0],
        -1,
        streaming_attn_output.size(-2),
        streaming_attn_output.size(-1),
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen, num_heads / sp_size, head_dim)
        # -> (1, total_seqlen / sp_size, num_heads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
        streaming_attn_output = gather_heads_scatter_seq(
            streaming_attn_output, seq_dim=1, head_dim=2
        )

    return attn_output, streaming_attn_output


def _mixed_attention_forward_func(
    query_states,
    key_states,
    value_states,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    streaming_mask,
    head_mask_type,
    causal,
):
    query = query_states.transpose(1, 2)
    key = key_states.transpose(1, 2)
    value = value_states.transpose(1, 2)

    query = query.reshape(-1, query.size(-2), query.size(-1))
    key = key.reshape(-1, key.size(-2), key.size(-1))
    value = value.reshape(-1, value.size(-2), value.size(-1))

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )  # [*, n_head/n, head_dim]
    attn_output = attn_output.view(
        query_states.shape[0], -1, attn_output.size(-2), attn_output.size(-1)
    )

    streaming_attn_output = block_streaming_attn_func(
        query,
        key,
        value,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k_=max_seqlen_q,
        max_seqlen_q_=max_seqlen_k,
        p_dropout=dropout_p,
        streaming_info=streaming_mask,
        head_mask_type=head_mask_type,
        softmax_scale=softmax_scale,
        is_causal=causal,
    )  # [*, n_head/n, head_dim]
    streaming_attn_output = streaming_attn_output.view(
        query_states.shape[0],
        -1,
        streaming_attn_output.size(-2),
        streaming_attn_output.size(-1),
    )

    return attn_output, streaming_attn_output
