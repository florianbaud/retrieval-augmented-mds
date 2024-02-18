import torch
import torch.nn as nn
import torch.nn.functional as F
import rich

from typing import List, Optional, Tuple, Union
from transformers.models.led.modeling_led import (
    _prepare_4d_attention_mask_inverted,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.activations import ACT2FN


class LEDDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.beta = nn.Parameter(torch.ones(1))
        self.beta_bias = nn.Parameter(torch.zeros(1))

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        ##### Bias attention weights #####
        attn_weights += (
            self.beta * attention_bias.view(attention_bias.shape[0], 1, -1)
            + self.beta_bias
        )
        ##################################

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class CopyDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config.d_model

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = LEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.skip_residual = config.skip_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """Forward with only Cross attention and Feed Forward"""

        present_key_value = ()

        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                attention_bias=attention_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )

            ############# HACK ##############
            self_attn_weights = hidden_states
            #################################

            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = (
            residual + hidden_states if not self.skip_residual else hidden_states
        )
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CopyDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [CopyDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        # self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        attention_bias=None,
        global_attention_mask=None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds: torch.Tensor = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        past_key_values_length = 0

        combined_attention_mask = None
        # if input_shape[-1] > 1:
        #     combined_attention_mask = _make_causal_mask(
        #         input_shape,
        #         inputs_embeds.dtype,
        #         past_key_values_length=past_key_values_length
        #     ).to(inputs_embeds.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = (
                combined_attention_mask
                + _prepare_4d_attention_mask_inverted(
                    attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask_inverted(
                encoder_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    # logger.warning(
                    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    attention_bias,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    attention_bias=attention_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,  # hidden_states before residual and ffn
            cross_attentions=all_cross_attentions,  # attn_weigths
        )


class DecoderForCopyGeneration(nn.Module):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.args = args
        self.config = config

        self.gates_attn = self.args.gates_mode

        embed_dim = config.d_model

        if self.gates_attn in ["nmt", "both"]:
            diverter_dim = embed_dim * 2
        elif self.gates_attn == "onlycopy":
            diverter_dim = embed_dim

        self.decoder = CopyDecoder(config)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.diverter = nn.Linear(diverter_dim, 2)
        # self.output_projection = nn.Linear(
        #     embed_dim, config.num_embeddings, bias=False)

        self.output_attention = True
        self.return_dict = True

        self.apply(self._init_weights)

    def forward(
        self,
        copy_sequence: torch.Tensor,
        decoder_hidden_states: torch.Tensor,  # Decoder hidden_states
        attention_bias: torch.Tensor = None,  # Matching score
        decoder_attention_mask: torch.Tensor = None,  # Decoder attention_mask
        encoder_hidden_states: torch.Tensor = None,  # Memory hidden_states
        encoder_attention_mask: torch.Tensor = None,  # Memory attention_mask
    ):
        # # batch_size x (seq_len * mips_topk)
        # print("Copy sequence shape :", copy_sequence.shape)
        # # batch_size x (tgt_len - 1) x hidden_size
        # print("Decoder hidden_states shape :", decoder_hidden_states.shape)
        # # batch_size x (mips_topk * hidden_size)
        # print("Matching score shape :", attention_bias.shape)
        # # batch_size x tgt_len
        # print("Decoder attention_mask shape :", decoder_attention_mask.shape)
        # # batch_size x (seq_len * mips_topk) x hidden_size
        # print("Memory hidden_states shape :", encoder_hidden_states.shape)
        # # batch_size x (seq_len * mips_topk)
        # print("Memory attention_mask shape :", encoder_attention_mask.shape)

        copy_decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            attention_bias=attention_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=decoder_hidden_states,
            use_cache=None,
            output_attentions=self.output_attention,
            output_hidden_states=None,
            return_dict=self.return_dict,
        )

        attn: torch.Tensor = copy_decoder_outputs.attentions[-1]
        outs: torch.Tensor = copy_decoder_outputs.last_hidden_state
        alignment_weight: torch.Tensor = copy_decoder_outputs.cross_attentions[-1]

        # # batch_size x (seq_len - 1) x hidden_size
        # print("Raw attention shape :", attn.shape)
        # # batch_size x (seq_len - 1) x hidden_size
        # print("Copy decoder outputs shape :", outs.shape)
        # # batch_size x 1 x (seq_len - 1) x memory_len
        # print("Attention weights shape :", alignment_weight.shape)
        # # batch_size x (seq_len - 1) x hidden_size
        # print("Decoder hidden states shape :", decoder_hidden_states.shape)

        alignment_weight = alignment_weight.squeeze(1)

        if self.gates_attn == "nmt":
            attn_normalized = self.alignment_layer_norm(attn)
            gates_input = torch.cat([outs, attn_normalized], -1)
        elif self.gates_attn == "onlycopy":
            gates_input = outs
        elif self.gates_attn == "both":
            gates_input = torch.cat([outs, decoder_hidden_states], -1)

        gates = F.softmax(self.diverter(gates_input), -1)
        gen_gate, copy_gate = gates.chunk(2, dim=-1)

        copy_probs = copy_gate * alignment_weight

        return gen_gate, copy_gate, copy_probs

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
