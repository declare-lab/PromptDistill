# transformers.__version__ == '4.43.3'
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .gem_filter_utils import find_context
logger = logging.get_logger(__name__)


class Qwen2SelectAttention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.reset()
        self.topk = 1024
        self.select_layer_idx = 19
        self.select_mode = False

    def reset(self):
        self.indecies = None
        return
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # [GemFilter] update below
        if self.select_mode:
            self.reset()
            find_context(self, query_states, key_states)

        if not self.select_mode and past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # [GemFilter] update above

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.LongTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     # Phi3FlashAttention2 attention does not support output_attentions

    #     output_attentions = False

    #     bsz, q_len, _ = hidden_states.size()

    #     qkv = self.qkv_proj(hidden_states)
    #     query_pos = self.num_heads * self.head_dim
    #     query_states = qkv[..., :query_pos]
    #     key_states = qkv[..., query_pos: query_pos +
    #                      self.num_key_value_heads * self.head_dim]
    #     value_states = qkv[..., query_pos +
    #                        self.num_key_value_heads * self.head_dim:]

    #     query_states = query_states.view(
    #         bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     key_states = key_states.view(
    #         bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     value_states = value_states.view(
    #         bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    #     kv_seq_len = key_states.shape[-2]
    #     if past_key_value is not None:
    #         kv_seq_len += past_key_value.get_usable_length(
    #             kv_seq_len, self.layer_idx)

    #     # Because the input can be padded, the absolute sequence length depends on the max position id.
    #     rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    #     cos, sin = self.rotary_emb(
    #         value_states, position_ids, seq_len=rotary_seq_len)

    #     query_states, key_states = apply_rotary_pos_emb(
    #         query_states, key_states, cos, sin, position_ids)
        
    #     # [GemFilter] update below
    #     if self.select_mode:
    #         self.reset()
    #         find_context(self, query_states, key_states)

    #     if not self.select_mode and past_key_value is not None:
    #         # Activate slicing cache only if the config has a value `sliding_windows` attribute
    #         cache_has_contents = past_key_value.get_seq_length(
    #             self.layer_idx) > 0
    #         if (
    #             getattr(self.config, "sliding_window", None) is not None
    #             and kv_seq_len > self.config.sliding_window
    #             and cache_has_contents
    #         ):
    #             slicing_tokens = 1 - self.config.sliding_window

    #             past_key = past_key_value[self.layer_idx][0]
    #             past_value = past_key_value[self.layer_idx][1]

    #             past_key = past_key[:, :, slicing_tokens:, :].contiguous()
    #             past_value = past_value[:, :, slicing_tokens:, :].contiguous()

    #             if past_key.shape[-2] != self.config.sliding_window - 1:
    #                 raise ValueError(
    #                     f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
    #                     f" {past_key.shape}"
    #                 )

    #             if attention_mask is not None:
    #                 attention_mask = attention_mask[:, slicing_tokens:]
    #                 attention_mask = torch.cat(
    #                     [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

    #         # Specific to RoPE models
    #         cache_kwargs = {"sin": sin, "cos": cos,
    #                         "cache_position": cache_position}
    #         key_states, value_states = past_key_value.update(
    #             key_states, value_states, self.layer_idx, cache_kwargs)
    #     # [GemFilter] update above
        
    #     # repeat k/v heads if n_kv_heads < n_heads
    #     key_states = repeat_kv(key_states, self.num_key_value_groups)
    #     value_states = repeat_kv(value_states, self.num_key_value_groups)

    #     attn_output, attn_weights = self.flash_softmax(
    #         query_states, key_states, value_states, attention_mask, q_len, position_ids)
    #     attn_output = attn_output.reshape(
    #         bsz, q_len, self.hidden_size).contiguous()

    #     attn_output = self.o_proj(attn_output)

    #     if not output_attentions:
    #         attn_weights = None

    #     return attn_output, attn_weights, past_key_value


    # def flash_softmax(self, query_states, key_states, value_states, attention_mask, q_len, position_ids):
    #     query_states = query_states.transpose(1, 2)
    #     key_states = key_states.transpose(1, 2)
    #     value_states = value_states.transpose(1, 2)
    #     input_dtype = query_states.dtype
    #     if input_dtype == torch.float32:
    #         query_states = query_states.to(torch.float16)
    #         key_states = key_states.to(torch.float16)
    #         value_states = value_states.to(torch.float16)

    #     attn_output = _flash_attention_forward(
    #         query_states,
    #         key_states,
    #         value_states,
    #         attention_mask,
    #         q_len,
    #         dropout=0.0,
    #         sliding_window=getattr(self, "sliding_window", None),
    #         use_top_left_mask=self._flash_attn_uses_top_left_mask,
    #         is_causal=True,
    #     )
    #     if input_dtype == torch.float32:
    #         attn_output = attn_output.to(torch.float32)
    #     return attn_output, None
