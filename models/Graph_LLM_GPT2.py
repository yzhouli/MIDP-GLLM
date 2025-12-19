import torch
from torch import nn
from typing import Callable, Optional, Union
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Block, GPT2Attention, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def create_causal_mask(seq_len, device='cpu'):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask == 0
    return mask.bool().to(device)

def create_batch_causal_mask(batch_size, seq_len, device='cpu'):
    mask = create_causal_mask(seq_len, device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, 1, seq_len, seq_len)
    return mask

@torch.no_grad()
class BatchMaskCache:
    """掩码缓存类，适用于固定尺寸的重复计算"""

    def __init__(self):
        self.mask_cache = {}

    @torch.no_grad()
    def get_mask(self, batch_size, seq_len, keep_rows, device):
        key = (batch_size, seq_len, keep_rows, device)

        if key not in self.mask_cache:
            # 创建坐标网格
            row_indices = torch.arange(seq_len, device=device).view(1, -1, 1)
            col_indices = torch.arange(seq_len, device=device).view(1, 1, -1)

            # 向量化条件
            mask = (row_indices < keep_rows) & (col_indices <= row_indices)
            mask = mask.expand(batch_size, -1, -1)

            self.mask_cache[key] = mask

        return self.mask_cache[key]

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

class Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, batch_adj_mask=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.batch_adj_mask = batch_adj_mask

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_value is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_value is not None and not is_cross_attention) or (
            past_key_value is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            if ('adj_graph_mask' not in kwargs.keys()) or (kwargs['adj_graph_mask'] is None):
                raise ValueError('graph_mask is not defined.')
            graph_mask, mask_cache = kwargs['adj_graph_mask'], self.batch_adj_mask
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
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )
                attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
                attn_output = attn_output[:, i, :]
                attn_output_li.append(attn_output)
            attn_output = torch.stack(attn_output_li, dim=1)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights

class Block(GPT2Block):
    def __init__(self, config, layer_idx=None, batch_adj_mask=None):
        super().__init__(config, layer_idx)
        self.attn = Attention(config=config, layer_idx=layer_idx, batch_adj_mask=batch_adj_mask)


class GraphLLM_GPT2(GPT2Model):

    def __init__(self, config, graph_layer_num):
        config.hidden_size = config.n_embd
        super().__init__(config)
        self.batch_adj_mask = BatchMaskCache()
        layer_num = config.num_hidden_layers - graph_layer_num
        blocks = [GPT2Block(config, layer_idx=i) for i in range(layer_num)]
        graph_blocks = [Block(config, layer_idx=i + layer_num, batch_adj_mask=self.batch_adj_mask) for i in range(graph_layer_num)]
        blocks.extend(graph_blocks)
        self.h = nn.ModuleList(blocks)


