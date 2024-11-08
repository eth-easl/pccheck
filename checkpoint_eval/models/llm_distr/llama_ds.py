# code from https://github.com/CoinCheung/gdGPT/blob/master/models/llama.py


import math
import os.path as osp
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import *
from transformers.models.llama.modeling_llama import _make_causal_mask, _expand_mask
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def init_weights(model, std):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaAttentionFlashAttn(LlamaAttention):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.inv_norm_factor = 1. / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.qkv_attn_func(query_states, key_states, value_states, attention_mask)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def qkv_attn_func(self, query_states, key_states, value_states, attention_mask):

        ## Now: qkv are [bs, num_heads, q_len, head_dim]
        ## flash-atten requires them to be: [bs, q_len, num_heads, head_dim]
        query_states = torch.einsum('bhld->blhd', query_states)
        key_states = torch.einsum('bhld->blhd', key_states)
        value_states = torch.einsum('bhld->blhd', value_states)

        attn_output = flash_attn_func(query_states, key_states,
                value_states, dropout_p=0., softmax_scale=self.inv_norm_factor,
                causal=True)

        attn_output = attn_output.flatten(2)
        return attn_output


class LlamaAttentionFast(LlamaAttentionFlashAttn):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def qkv_attn_func(self, query_states, key_states, value_states, attention_mask):

        ## Now: qkv are [bs, num_heads, q_len, head_dim]

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states,
                    value_states, dropout_p=0., attn_mask=attention_mask,
                    scale=self.inv_norm_factor)
        attn_output = torch.einsum('bhld->blhd', attn_output)

        bsz, _, q_len, _ = query_states.size()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return attn_output



class LlamaDecoderLayerTupleIO(LlamaDecoderLayer):

    def __init__(self, config: LlamaConfig, res=None, ind=0,
            gradient_checkpointing=False, use_flash_attn=False):
        super().__init__(config)
        init_weights(self, config.initializer_range)
        self.load_state_dict(res[ind], strict=False)
        self.gradient_checkpointing = gradient_checkpointing
        if use_flash_attn: self.self_attn = LlamaAttentionFlashAttn(config=config)
        #  self.self_attn = LlamaAttentionFast(config=config)
        #  self.self_attn = LlamaAttention(config=config)

    @torch.compile
    def forward(self, inputs):
        """
        Args:
            inputs:
                hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                    `(batch, src_len)` where padding elements are 0
        """

        hidden_states, attention_mask = inputs
        batch_size, seq_length, _ = hidden_states.shape
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )
        # because original position_ids is None, we use this directly
        #  position_ids = torch.arange(
        #      0, seq_length, dtype=torch.long, device=hidden_states.device
        #  ).unsqueeze(0).view(-1, seq_length)
        position_ids = (attention_mask.cumsum(dim=-1) - 1) * attention_mask
        if self.gradient_checkpointing and self.training:
            outputs = torch.utils.checkpoint.checkpoint(
                super().forward,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                padding_mask=attention_mask,
                position_ids=position_ids
            )
        else:
            outputs = super().forward(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    padding_mask=attention_mask,
                    position_ids=position_ids)
        hidden_states = outputs[0]

        return hidden_states, attention_mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class LlamaEnter(nn.Module):
    """
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, res=None, ind=0):
        super(LlamaEnter, self).__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                config.hidden_size, config.pad_token_id)

        init_weights(self, config.initializer_range)
        self.load_state_dict(res[ind])

    @torch.compile
    def forward(self, inputs):
        output_attentions = False
        output_hidden_states = False
        use_cache = False
        return_dict = False
        inputs_embeds = None
        past_key_values = None
        position_ids = None
        input_ids = inputs[..., 0].contiguous()
        attention_mask = inputs[..., 1].contiguous()

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool,
                device=inputs_embeds.device
            )
        attention_mask = attention_mask.clone()

        hidden_states = inputs_embeds
        return hidden_states, attention_mask

    @property
    def weight(self):
        return self.embed_tokens.weight


class LlamaExit(nn.Module):
    """
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, res=None, ind=0):
        super(LlamaExit, self).__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,
                config.hidden_size, config.pad_token_id)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        init_weights(self, config.initializer_range)
        self.load_state_dict(res[ind])

    @torch.compile
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        hidden_states = self.norm(hidden_states)
        logits = F.linear(hidden_states, self.embed_tokens.weight, None)
        return logits

    @property
    def weight(self):
        return self.embed_tokens.weight



## llama does not tie weights
def get_llama_causal_lm_specs(config, res={}, grad_ckpt=False,
        tie_emb=False, use_flash_attn=False, from_scratch=False):
    specs = []
    specs.append(LlamaEnter(config, res=res, ind=0))

    for i in range(1, config.num_hidden_layers+1):
        specs.append(LayerSpec(LlamaDecoderLayerTupleIO, config,
            res=res, ind=i))

    ind = config.num_hidden_layers + 1
    specs.append(LlamaExit(config, res=res, ind=ind))
    return specs
