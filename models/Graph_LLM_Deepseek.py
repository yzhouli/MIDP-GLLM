import torch
from torch import nn
from typing import Callable, Optional
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer, Qwen2Attention, \
    apply_rotary_pos_emb, eager_attention_forward

def create_causal_mask(seq_len, device='cpu'):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask == 0
    return mask.bool().to(device)

def create_batch_causal_mask(batch_size, seq_len, device='cpu'):
    mask = create_causal_mask(seq_len, device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, 1, seq_len, seq_len)
    return mask

def compute_mask(rel_type, attention_mask, graph_mask, device='cpu'):
    if attention_mask is None:
        B, N, N = graph_mask.shape
        attention_mask = create_batch_causal_mask(batch_size=B, seq_len=N, device=device)
    if rel_type is None:
        attention_mask = graph_mask.unsqueeze(1)
    elif rel_type == 'causal_only':
        attention_mask = attention_mask
    elif rel_type == 'causal_and_adj_bool':
        graph_mask = graph_mask.unsqueeze(1).bool()
        attention_mask = attention_mask & graph_mask
    elif rel_type == 'multi_scale_causal_adj':
        if type(graph_mask) is torch.bool:
            attention_mask = attention_mask | graph_mask.unsqueeze(1)
        else:
            attention_mask = attention_mask + graph_mask.unsqueeze(1)
    return attention_mask

class Attention(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
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
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if ('enable_graph_mask' in kwargs.keys()) and kwargs['enable_graph_mask']:
            if ('graph_mask' not in kwargs.keys()) or (kwargs['graph_mask'] is None):
                raise ValueError('graph_mask is not defined.')
            graph_mask, mask_cache = kwargs['graph_mask'], kwargs['batch_adj_mask']
            graph_mask_li = [mask_cache.get_mask(graph_mask.size(0), graph_mask.size(-1), i, attention_mask.device)
                             for i in range(graph_mask.size(-1))]
            attn_output_li = []
            for i in range(graph_mask.size(-1)):
                graph_mask = graph_mask_li[i]
                rel_type = None
                if 'rel_type' in kwargs.keys():
                    rel_type = kwargs['rel_type']
                attention_mask = compute_mask(rel_type=rel_type, attention_mask=attention_mask,
                                              graph_mask=graph_mask, device=query_states.device)
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,  # main diff with Llama
                    **kwargs,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output[:, i, :, :]
                attn_output_li.append(attn_output)
            attn_output = torch.stack(attn_output_li, dim=1)
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # main diff with Llama
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()


        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Block(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Attention(config=config, layer_idx=layer_idx)

class GraphLLM_Deepseek(Qwen2Model):
    def __init__(self, config, graph_layer_num):
        super().__init__(config)
        layer_num = config.num_hidden_layers - graph_layer_num
        blocks = [Qwen2DecoderLayer(config, layer_idx=i) for i in range(layer_num)]
        graph_blocks = [Block(config, layer_idx=i + layer_num) for i in range(graph_layer_num)]
        blocks.extend(graph_blocks)
        self.layers = nn.ModuleList(blocks)