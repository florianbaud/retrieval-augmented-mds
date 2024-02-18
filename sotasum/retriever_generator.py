import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import rich

from .model_config import ModelConfig
from .decoder import CopyTokenDecoder
from .mips import Mips, MipsModelOutput
from .decoder_own import DecoderForCopyGeneration
from adapters import AutoAdapterModel
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    LEDConfig,
    AutoTokenizer,
    LongformerModel,
    LongformerConfig,
    LongformerTokenizer,
)
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig


@dataclass
class RGEncoderModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
    memory: torch.FloatTensor = None
    memory_mask: torch.FloatTensor = None
    memory_bias: torch.FloatTensor = None
    copy_sequence: torch.FloatTensor = None
    mips_scores: torch.FloatTensor = None
    examples: list = None
    faiss_scores: np.ndarray = None
    query_cls: np.ndarray = None


@dataclass
class RGDecoderModelOutput(ModelOutput):
    logits: torch.FloatTensor
    copy_probs: torch.FloatTensor = None
    copy_gate: torch.FloatTensor = None
    past_key_values: Tuple[torch.FloatTensor] = None
    decoder_attentions: torch.FloatTensor = None
    cross_attentions: torch.FloatTensor = None


class SotasumEncoder(nn.Module):
    def __init__(
        self,
        args: ModelConfig = ModelConfig(),
        encoder=None,
        doc_sep_id: int = None,
    ) -> None:
        super().__init__()

        self.args = args
        self.encoder = encoder
        self.doc_sep_id = doc_sep_id
        self.main_input_name = self.encoder.main_input_name

        # self.query_tokenizer = LongformerTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=self.args.query_encoder_path,
        # )
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            self.args.query_encoder_path
        )

        if not self.args.mips_disabled:
            self.mips = Mips(args=args)

            # query_config = LongformerConfig.from_pretrained(
            #     pretrained_model_name_or_path=self.args.query_encoder_path,
            #     gradient_checkpointing=self.args.gradient_checkpointing,
            # )
            # self.query_encoder = LongformerModel.from_pretrained(
            #     pretrained_model_name_or_path=self.args.query_encoder_path,
            #     config=query_config,
            # )

            self.query_encoder = AutoAdapterModel.from_pretrained(
                self.args.query_encoder_path
            )
            self.query_encoder.load_adapter(
                "allenai/specter2", source="hf", load_as="specter2", set_active=True
            )

            if self.args.query_state_dict is not None:
                query_state_dict = torch.load(self.args.query_state_dict)
                self.query_encoder.load_state_dict(query_state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        query_input_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        mips_ignore_indexes: list = None,
        aid: list = None,
        aid_counts: torch.Tensor = None,
        target_str: list = None,
        input_str: list = None,
        return_dict: bool = True,
        **kwargs,
    ):
        memory, memory_bias, memory_mask, copy_seq, mips_scores = (
            None,
            None,
            None,
            None,
            None,
        )
        mips_out = MipsModelOutput()
        if not self.args.mips_disabled:
            query_batch_size = query_input_ids.shape[0]

            # put global attention on <s> token
            # query_global_attention_mask = torch.zeros_like(
            #     query_input_ids, device=query_input_ids.device
            # )
            # query_global_attention_mask[:, 0] = 1

            # if not self.args.use_attention_mask:
            #     query_attention_mask = None
            query_encoder_outputs = self.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                # global_attention_mask=query_global_attention_mask,
            )

            query = query_encoder_outputs[0]
            # Implement layer norm ? (LongFormer has layer norm)
            # query = _layer_norm(query)
            query: torch.Tensor = query[:, :1, :]

            mips_query = query[:, 0, :].detach().cpu().float().numpy()

            mips_out: MipsModelOutput = self.mips(
                queries=mips_query,
                aid=aid,
                aid_counts=aid_counts,
                target_str=target_str,
                input_str=input_str,
                k=self.args.mips_topk,
                ignore_indexes=mips_ignore_indexes,
            )

            if mips_out.metrics is not None:
                kwargs.get("logger")(mips_out.metrics, sync_dist=True)

            mips_cls = mips_out.mips_last_hidden_state[:, :, 0, :]
            mips_scores = (query @ mips_cls.transpose(1, 2)).squeeze(1)

            with torch.no_grad():
                query_norms: torch.Tensor = torch.norm(
                    query,
                    dim=2,
                    keepdim=True,
                )
                mips_norms: torch.Tensor = torch.norm(
                    mips_cls,
                    dim=2,
                    keepdim=True,
                )
            mips_scores /= (query_norms * mips_norms).squeeze(2)

            ### Approx cosine similarity from https://aclanthology.org/2021.acl-long.567/ ??? ###
            ### => https://github.com/jcyk/copyisallyouneed ###
            # mips_scores /= self.mips.max_norm**2

            # print('MIPS Recomputed scores :', mips_scores)

            memory_last_hidden_state: torch.Tensor = mips_out.memory_outputs[0]
            memory_seq_len = memory_last_hidden_state.shape[1]
            topk = mips_out.mips_last_hidden_state.shape[1]

            memory = memory_last_hidden_state.reshape(
                query_batch_size, memory_seq_len * topk, -1
            )
            memory_mask = mips_out.memory_attention_mask.view(query_batch_size, -1)
            memory_bias = (
                mips_scores.unsqueeze(-1)
                .expand(-1, -1, memory_seq_len)
                .reshape(query_batch_size, -1)
            )
            copy_seq = mips_out.memory_input_ids.view(query_batch_size, -1)

        # put global attention on <s> token
        global_attention_mask = torch.zeros_like(input_ids, device=input_ids.device)
        global_attention_mask[:, 0] = 1

        # Put global attetion on <doc> token
        doc_token = input_ids == self.doc_sep_id
        global_attention_mask[doc_token] = 1

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=return_dict,
        )

        if return_dict:
            return RGEncoderModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                global_attentions=encoder_outputs.global_attentions,
                memory=memory,
                memory_mask=memory_mask,
                memory_bias=memory_bias,
                copy_sequence=copy_seq,
                mips_scores=mips_scores,
                examples=mips_out.examples,
                faiss_scores=mips_out.scores,
                query_cls=mips_out.query_cls,
            )
        return encoder_outputs + (memory, memory_mask, memory_bias, copy_seq)


class RetrieverGenerator(PreTrainedModel):
    def __init__(self, args: ModelConfig = ModelConfig()) -> None:
        super().__init__(config=PretrainedConfig())
        self.args = args

        self.tokenizer: LEDTokenizer = LEDTokenizer.from_pretrained(
            args.model_name,
        )
        _ = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.args.doc_sep]}
        )

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids(self.args.doc_sep)

        self.config = LEDConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            use_cache=False,
            gradient_checkpointing=self.args.gradient_checkpointing,
        )

        self.model = LEDForConditionalGeneration.from_pretrained(
            args.model_name,
            config=self.config,
        )

        # Model embeddings matrix update.
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.main_input_name = self.model.main_input_name

        self.encoder = SotasumEncoder(
            args=args,
            encoder=self.model.get_encoder(),
            doc_sep_id=self.docsep_token_id,
        )

        if not self.args.mips_disabled:
            if self.args.use_own_decoder:
                copy_config = LEDConfig(
                    use_cache=False,
                    d_model=self.config.d_model,
                    decoder_attention_heads=1,
                    decoder_layers=self.args.copy_decoder_layers,
                    num_embeddings=self.model.led.shared.num_embeddings,
                    skip_residual=self.args.skip_residual,
                )
                self.copy_decoder = DecoderForCopyGeneration(
                    args=self.args,
                    config=copy_config,
                )
            else:
                embed_dim = 1024
                self.decoder_head = CopyTokenDecoder(
                    vocabs=None,
                    tgt_embed=self.model.led.shared,
                    label_smoothing=None,
                    embed_dim=embed_dim,
                    ff_embed_dim=2048,
                    dropout=0.1,
                )

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.FloatTensor,
        encoder_outputs: RGEncoderModelOutput,
        ########### HACK ###########
        query_input_ids=None,
        query_attention_mask=None,
        input_str=None,
        ############################
        **kwargs,
    ):
        use_cache = kwargs.get("use_cache", None)
        past_key_values = kwargs.get("past_key_values", None)

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if self.generation_config.num_beams > 1 and not self.args.mips_disabled:
            expand_size = self.generation_config.num_beams
            batch_size = input_ids.shape[0] // expand_size

            index = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, expand_size)
                .view(-1)
                .to(self.model.device)
            )

            encoder_outputs.memory = encoder_outputs.memory.index_select(0, index)
            encoder_outputs.memory_bias = encoder_outputs.memory_bias.index_select(
                0, index
            )
            encoder_outputs.memory_mask = encoder_outputs.memory_mask.index_select(
                0, index
            )
            encoder_outputs.copy_sequence = encoder_outputs.copy_sequence.index_select(
                0, index
            )

        model_input = {
            "input_ids": input_ids,
            "encoder_attention_mask": kwargs.get("attention_mask", None),
            "encoder_outputs": encoder_outputs,
            "use_cache": use_cache,
            "past_key_values": past_key_values,
        }
        return model_input

    def _reorder_cache(self, past, beam_idx):
        return self.model._reorder_cache(past, beam_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_outputs: RGEncoderModelOutput = None,
        encoder_attention_mask: torch.Tensor = None,
        return_dict: bool = True,
        use_cache: tuple = None,
        past_key_values: torch.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        encoder_memory = encoder_outputs.memory
        encoder_memory_bias = encoder_outputs.memory_bias
        encoder_memory_mask = encoder_outputs.memory_mask
        encoder_copy_sequence = encoder_outputs.copy_sequence

        decoder_outputs = self.model(
            input_ids=None,
            attention_mask=encoder_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=input_ids,
            decoder_attention_mask=None,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        copy_probs, copy_gate, copy_probs_sentence = None, None, None
        if not self.args.mips_disabled:
            decoder_hidden_states = decoder_outputs.decoder_hidden_states[-1]
            if self.args.use_own_decoder:
                gen_gate, copy_gate, copy_probs = self.copy_decoder(
                    copy_sequence=encoder_copy_sequence,
                    decoder_hidden_states=decoder_hidden_states,  # Decoder hidden_states
                    attention_bias=encoder_memory_bias,  # Matching score
                    decoder_attention_mask=attention_mask,  # Decoder attention_mask
                    encoder_hidden_states=encoder_memory,  # Memory hidden_states
                    encoder_attention_mask=encoder_memory_mask,
                )

                probs: torch.Tensor = gen_gate * F.softmax(decoder_outputs.logits, -1)

                bsz, seq_len, _ = decoder_hidden_states.size()
                index = encoder_copy_sequence.reshape((bsz, 1, -1)).expand(
                    -1, seq_len, -1
                )
                probs_both = probs.scatter_add_(-1, index, copy_probs)

                if output_attentions:
                    copy_probs_sentence = torch.zeros_like(probs).scatter_add_(
                        -1, index, copy_probs
                    )

                outs = torch.log(probs_both + 1e-7)
            else:
                decoder_hidden_states = decoder_hidden_states.transpose(0, 1)
                encoder_memory = encoder_memory.reshape(
                    -1,
                    encoder_memory.shape[0] * self.args.mips_topk,
                    encoder_memory.shape[2],
                )
                encoder_memory_mask = ~encoder_memory_mask.transpose(0, 1)
                encoder_memory_bias = encoder_memory_bias.transpose(0, 1)
                outs = self.decoder_head(
                    outs=decoder_hidden_states,
                    mem=encoder_memory,
                    mem_mask=encoder_memory_mask,
                    mem_bias=encoder_memory_bias,
                    copy_seq=encoder_copy_sequence,
                    data=None,
                    work=None,
                ).transpose(0, 1)
        else:
            outs = decoder_outputs.logits

        if return_dict:
            return RGDecoderModelOutput(
                logits=outs,
                copy_gate=copy_gate,
                copy_probs=copy_probs,
                past_key_values=decoder_outputs.past_key_values,
                decoder_attentions=copy_probs,
                cross_attentions=copy_probs_sentence,
            )

        return outs
