import torch
from typing import Callable, Optional, Union
from torch.nn import RMSNorm
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Attention, Qwen3MLP, Qwen3RMSNorm, \
    Qwen3Model, Qwen3RotaryEmbedding, Qwen3ForCausalLM
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging, is_torch_xpu_available
from transformers.utils.generic import check_model_inputs, GeneralInterface
from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.integrations.eager_paged import eager_paged_attention_forward
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.integrations.flash_paged import paged_attention_forward
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.integrations.sdpa_paged import sdpa_attention_paged_forward


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

logger = logging.get_logger(__name__)

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()

def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 2.xpu
    #   - torch version >= 2.8
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8 and not isinstance(key, torch.fx.Proxy)
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and not isinstance(key, torch.fx.Proxy)

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

def compute_mask(rel_type, attention_mask, graph_mask):
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

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if ('enable_graph_mask' in kwargs.keys()) and kwargs['enable_graph_mask']:
        if ('graph_mask' not in kwargs.keys()) or (kwargs['graph_mask'] is None):
            raise ValueError('graph_mask is not defined.')
        graph_mask, mask_cache = kwargs['graph_mask'], kwargs['batch_adj_mask']
        graph_mask_li = [mask_cache.get_mask(graph_mask.size(0), graph_mask.size(-1), i, attention_mask.device) for i in range(graph_mask.size(-1))]
        attn_output_li = []
        for i in range(graph_mask.size(-1)):
            graph_mask = graph_mask_li[i]
            rel_type = None
            if 'rel_type' in kwargs.keys():
                rel_type = kwargs['rel_type']
            attention_mask = compute_mask(rel_type=rel_type, attention_mask=attention_mask, graph_mask=graph_mask)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout,
                scale=scaling,
                is_causal=is_causal,
                **sdpa_kwargs,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output[:, i, :, :]
            attn_output_li.append(attn_output)
        attn_output = torch.stack(attn_output_li, dim=1)
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

    if ('return_qk_states' in kwargs.keys()) and kwargs['return_qk_states']:
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if type(attention_mask) is torch.bool:
            attention_mask = torch.where(attention_mask, 1.0, 1e-6)
            attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return attn_output, attn_weights

    return attn_output, None

def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if kwargs['enable_graph_mask']:
        graph_mask = kwargs['graph_mask']
        graph_mask = graph_mask.unsqueeze(1)
        attn_weights = attn_weights + graph_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class HopTable(nn.Module):
    def __init__(self, hop_table_num, device='cpu', cut_off_threshold=-1e6):
        super().__init__()
        self.hop_table = nn.Parameter(torch.randn(hop_table_num), requires_grad=True).to(device)
        self.hop_table_num = hop_table_num
        self.cut_off_threshold = cut_off_threshold
        cut_off_table = [0 for i in range(self.hop_table_num)]
        cut_off_table[0] = self.cut_off_threshold
        self.cut_off_table = torch.tensor(cut_off_table, requires_grad=False).to(device)

    def forward(self, ids_mat):
        table = self.hop_table + self.cut_off_table
        return table[ids_mat]  # 示例操作

class AttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "paged_attention": paged_attention_forward,
        "sdpa": sdpa_attention_forward,
        "sdpa_paged": sdpa_attention_paged_forward,
        "eager_paged": eager_paged_attention_forward,
    }


# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

class CustomGraphAttention(Qwen3Attention):

    def __init__(self, config, layer_idx: int, batch_adj_mask, enable_graph_mask: bool):
        super().__init__(config=config, layer_idx=layer_idx)
        self.enable_graph_mask = enable_graph_mask
        self.batch_adj_mask=batch_adj_mask
        self.hop_table = None
        # self.config = config
        # self.layer_idx = layer_idx
        # self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # self.scaling = self.head_dim**-0.5
        # self.attention_dropout = config.attention_dropout
        # self.is_causal = True
        #
        # self.q_proj = nn.Linear(
        #     config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.k_proj = nn.Linear(
        #     config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.v_proj = nn.Linear(
        #     config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.o_proj = nn.Linear(
        #     config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        # )
        # self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        # self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        # self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

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

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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

        graph_mask = None
        if self.enable_graph_mask:
            if ('adj_graph_mask' not in kwargs.keys()) and ('hop_graph_mask' not in kwargs.keys()):
                raise ValueError(
                    'Graph mask not specified. Please select either the critical matrix (“adj_graph_mask”) or the K-hop shortest path matrix (‘hop_graph_mask’).')
            if 'hop_graph_mask' in kwargs.keys():
                if 'hop_table_num' not in kwargs.keys():
                    raise RuntimeError(
                        'K-hop graph encoding is used, please enter the parameter “hop_table_num” to determine the learning matrix.')
                hop_graph_mask = kwargs['hop_graph_mask']
                if self.hop_table is None:
                    self.hop_table = HopTable(hop_table_num=kwargs['hop_table_num'], device=hop_graph_mask.device)
                graph_mask = self.hop_table(hop_graph_mask)
            else:
                graph_mask = kwargs['adj_graph_mask']
            kwargs['batch_adj_mask'] = self.batch_adj_mask
            # graph_mask = graph_mask.bool()

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            enable_graph_mask=self.enable_graph_mask,
            graph_mask=graph_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CustomDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx, batch_adj_mask, enable_graph_mask: bool):
        super().__init__(config, layer_idx)
        # self.hidden_size = config.hidden_size
        #
        self.batch_adj_mask = batch_adj_mask
        self.self_attn = CustomGraphAttention(config=config, layer_idx=layer_idx, batch_adj_mask=batch_adj_mask, enable_graph_mask=enable_graph_mask)
        #
        # self.mlp = Qwen3MLP(config)
        # self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.attention_type = config.layer_types[layer_idx]

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, qk_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, qk_states

class GraphLLM(Qwen3Model):
    def __init__(self, config, graph_layer_num: int):
        super().__init__(config)
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size
        #
        self.batch_adj_mask = BatchMaskCache()
        if graph_layer_num < 0 or graph_layer_num > config.num_hidden_layers:
            raise ValueError(f"graph_layer_num ({graph_layer_num}) must be between 0 and config.num_hidden_layers")
        elif graph_layer_num == 0:
            print(
                f"\033[31m Lack of graph inference module, graph layer num: {graph_layer_num} < time sequence layer num: {self.config.num_hidden_layers}\033[0m")
        elif graph_layer_num > config.num_hidden_layers:
            raise ValueError(
                f'graph layer num: {graph_layer_num} > time sequence layer num: {self.config.num_hidden_layers}')
        elif graph_layer_num == config.num_hidden_layers:
            print(
                f"\033[31m Lack of time inference module, graph layer num: {graph_layer_num} > time sequence layer num: {self.config.num_hidden_layers}\033[0m")
        time_layer_num = config.num_hidden_layers - graph_layer_num
        time_layers = [CustomDecoderLayer(config, layer_idx, batch_adj_mask=self.batch_adj_mask, enable_graph_mask=False) for layer_idx in range(time_layer_num)]
        graph_layer = [CustomDecoderLayer(config, layer_idx, batch_adj_mask=self.batch_adj_mask, enable_graph_mask=True) for layer_idx in range(graph_layer_num)]
        time_layers.extend(graph_layer)
        self.embed_tokens = None
        self.layers = nn.ModuleList(time_layers)
        self.qk_states_li = [i for i in range(len(time_layers))]
        # self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        # self.gradient_checkpointing = False
        # self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        #
        # # Initialize weights and apply final processing
        # self.post_init()

    @torch.no_grad()
    def get_qk_states(self):
        return self.qk_states_li

    @check_model_inputs
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, qk_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            self.qk_states_li[index] = qk_states

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
