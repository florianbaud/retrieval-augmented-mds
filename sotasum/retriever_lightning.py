from typing import Any
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import faiss
import rich

from .data_loaders import (
    PretrainMultiXScienceDataset,
    PretrainAbstractMultiXScienceDataset,
    load_mips_multi_x_science,
)
from dataclasses import dataclass
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import (
    get_linear_schedule_with_warmup,
    LongformerModel,
    LongformerConfig,
    LongformerTokenizer,
)


@dataclass
class RetrieverConfig:
    query_pretrained_model_name_or_path: str = "allenai/longformer-large-4096"
    mips_pretrained_model_name_or_path: str = "allenai/longformer-large-4096"
    mips_state_dict: str = None
    query_state_dict: str = None
    inner_product: bool = False
    temperature: float = 1.0
    learning_rate: float = 1e-03
    gradient_checkpointing: bool = False
    batch_size: int = 1
    validation_batch_size: int = 2
    label_smoothing: float = 0.1
    warmup_steps: int = 500
    pooling: bool = False
    token_loss: bool = False
    top_k: int = 5
    dry_run: bool = False
    test_full_data: bool = False
    save_models_dir: str = None
    mips_tok_max_length: int = 1024
    knowledge_base_column: str = (
        "ref_abstract"  # "related_work", "ref_abstract" for MXS
    )
    knowledge_base_path: str = None


def get_phi(xb: np.ndarray):
    return (xb**2).sum(1).max()


def augment_xb(xb, phi=None):
    norms = (xb**2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((xb, extracol.reshape(-1, 1)))


def augment_xq(xq):
    extracol = np.zeros(len(xq), dtype="float32")
    return np.hstack((xq, extracol.reshape(-1, 1)))


def retriever_metrics(pred: torch.Tensor, counts: torch.Tensor) -> dict:
    recall = (pred.sum(-1) / counts).mean().item()

    reciprocal_rank = 1 / pred.argmax(-1)
    mask = reciprocal_rank == torch.inf
    reciprocal_rank = reciprocal_rank.masked_fill(mask, 0).mean().item()

    precision = (pred.cumsum(-1) / torch.arange(1, pred.shape[-1] + 1)) * pred
    average_precision = (precision.sum(-1) / counts).mean().item()

    outputs = {
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "average_precision": average_precision,
    }

    return outputs


class RetrieverLightning(pl.LightningModule):
    def __init__(self, model_config: RetrieverConfig = RetrieverConfig()) -> None:
        super().__init__()
        self.model_config = model_config
        self.losses = []

        self.query_tokenizer = None
        self.query_encoder = None
        self.mips_tokenizer = None
        self.mips_encoder = None
        self.knowledge_base = None

    def setup(self, stage: str) -> None:
        self.query_tokenizer = LongformerTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_config.query_pretrained_model_name_or_path,
        )
        query_config = LongformerConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_config.query_pretrained_model_name_or_path,
            gradient_checkpointing=self.model_config.gradient_checkpointing,
        )
        self.query_encoder = LongformerModel.from_pretrained(
            pretrained_model_name_or_path=self.model_config.query_pretrained_model_name_or_path,
            config=query_config,
        )

        # _ = self.query_tokenizer.add_special_tokens(
        #     {"additional_special_tokens": [self.model_config.doc_sep]})
        # _ = self.query_encoder.resize_token_embeddings(
        #     len(self.query_tokenizer))
        # self.doc_sep_id = self.query_tokenizer.convert_tokens_to_ids(
        #     self.model_config.doc_sep)

        if self.model_config.query_state_dict is not None:
            query_state_dict = torch.load(self.model_config.query_state_dict)
            self.query_encoder.load_state_dict(query_state_dict)

        self.mips_tokenizer = LongformerTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_config.mips_pretrained_model_name_or_path,
        )
        mips_config = LongformerConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_config.mips_pretrained_model_name_or_path,
            gradient_checkpointing=self.model_config.gradient_checkpointing,
        )
        self.mips_encoder = LongformerModel.from_pretrained(
            pretrained_model_name_or_path=self.model_config.mips_pretrained_model_name_or_path,
            config=mips_config,
        )

        if self.model_config.mips_state_dict is not None:
            mips_state_dict = torch.load(self.model_config.mips_state_dict)
            self.mips_encoder.load_state_dict(mips_state_dict)

        if self.model_config.pooling:
            self.query_pooling = nn.Linear(
                in_features=query_config.hidden_size,
                out_features=128,
                bias=False,
            )

            self.mips_pooling = nn.Linear(
                in_features=mips_config.hidden_size,
                out_features=128,
                bias=False,
            )

        if self.model_config.token_loss:
            self.query_bow = BOWModel(
                self.query_encoder.embeddings.word_embeddings,
                tokenizer=self.query_tokenizer,
            )
            self.mips_bow = BOWModel(
                self.mips_encoder.embeddings.word_embeddings,
                tokenizer=self.mips_tokenizer,
            )
            # self.query_linear = nn.Linear(
            #     in_features=query_config.hidden_size,
            #     out_features=mips_config.vocab_size,
            #     bias=True,
            # )
            # self.mips_linear = nn.Linear(
            #     in_features=mips_config.hidden_size,
            #     out_features=query_config.vocab_size,
            #     bias=True,
            # )

        self.knowledge_base = load_mips_multi_x_science(
            data_path=self.model_config.knowledge_base_path,
            script_path="multi_x_science_sum",
            column=self.model_config.knowledge_base_column,
        )

    def forward(
        self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        query_global_attention_mask = torch.zeros_like(
            query_input_ids, device=self.device
        )
        query_global_attention_mask[:, 0] = 1
        # doc_token = query_input_ids == self.doc_sep_id
        # query_global_attention_mask[doc_token] = 1
        query_output = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            global_attention_mask=query_global_attention_mask,
        )
        return query_output

    def training_step(self, batch, batch_idx):
        self.train()

        query_output = self.forward(
            query_input_ids=batch["query_input_ids"],
            query_attention_mask=batch["query_attention_mask"],
        )
        query_cls: torch.Tensor = query_output[0][:, 0, :]

        mips_input_ids = batch["mips_input_ids"]
        mips_attention_mask = batch["mips_attention_mask"]
        mips_global_attention_mask = torch.zeros_like(
            mips_input_ids, device=self.device
        )
        mips_global_attention_mask[:, 0] = 1
        mips_output = self.mips_encoder(
            input_ids=mips_input_ids,
            attention_mask=mips_attention_mask,
            global_attention_mask=mips_global_attention_mask,
        )
        mips_cls: torch.Tensor = mips_output[0][:, 0, :]

        if self.model_config.pooling:
            query_cls = self.query_pooling(query_cls)
            mips_cls = self.mips_pooling(mips_cls)

        # query_cls = nn.functional.normalize(
        #     query_cls,
        # )
        # mips_cls = nn.functional.normalize(
        #     mips_cls,
        # )

        sentence_scores = (query_cls @ mips_cls.T) / self.model_config.temperature
        sentence_target = torch.arange(sentence_scores.shape[0], device=self.device)
        sentence_loss = nn.functional.cross_entropy(
            input=sentence_scores,
            target=sentence_target,
        )

        loss = sentence_loss

        loss_dict = {
            "loss": loss.item(),
        }

        if self.model_config.token_loss:
            query_loss = self.query_bow(
                mips_cls, batch["query_input_ids"], batch["query_attention_mask"]
            )
            mips_loss = self.mips_bow(query_cls, mips_input_ids, mips_attention_mask)
            # query_token: torch.Tensor = self.query_linear(query_cls)
            # mips_token: torch.Tensor = self.mips_linear(mips_cls)

            # query_mask = (mips_input_ids == self.mips_tokenizer.unk_token_id) | (mips_input_ids == self.mips_tokenizer.cls_token_id) | (
            #     mips_input_ids == self.mips_tokenizer.eos_token_id) | (~mips_attention_mask.bool())
            # mips_mask = (query_input_ids == self.query_tokenizer.unk_token_id) | (query_input_ids == self.query_tokenizer.cls_token_id) | (
            #     query_input_ids == self.query_tokenizer.eos_token_id) | (~query_attention_mask.bool())

            # query_token_scores = -(query_token.log_softmax(-1))
            # mips_token_scores = -(mips_token.log_softmax(-1))

            # query_token_loss = query_token_scores.gather(
            #     -1, mips_input_ids).masked_fill(query_mask, 0.)
            # mips_token_loss = mips_token_scores.gather(
            #     -1, query_input_ids).masked_fill(mips_mask, 0.)

            # query_token_loss = query_token_loss.sum(-1).mean()
            # mips_token_loss = mips_token_loss.sum(-1).mean()

            loss_dict["sentence_loss"] = loss_dict["loss"]
            token_loss = query_loss + mips_loss
            loss_dict["token_loss"] = token_loss.item()
            loss += token_loss
            loss_dict["loss"] = loss.item()

        with torch.no_grad():
            scores = F.normalize(query_cls) @ F.normalize(mips_cls).T
            _, i = scores.topk(1)
            truth = torch.arange(scores.shape[0]).to(self.device)
            loss_dict["train_accuracy"] = (i.view(-1) == truth).float().mean().item()

        self.log_dict(loss_dict, sync_dist=True)

        return loss

    @torch.inference_mode()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        query_output = self.forward(
            query_input_ids=batch["query_input_ids"],
            query_attention_mask=batch["query_attention_mask"],
        )
        query_cls: torch.Tensor = query_output[0][:, 0, :]

        outputs = dict()
        if getattr(self, "_full_data", None) == None:
            mips_global_attention_mask = torch.zeros_like(
                batch["mips_input_ids"], device=self.device
            )
            mips_global_attention_mask[:, 0] = 1
            mips_output = self.mips_encoder(
                input_ids=batch["mips_input_ids"],
                attention_mask=batch["mips_attention_mask"],
                global_attention_mask=mips_global_attention_mask,
            )
            mips_cls: torch.Tensor = mips_output[0][:, 0, :]

            scores = query_cls @ mips_cls.T
            _, i = scores.topk(1)

            outputs.update({"scores": scores})

            truth = torch.arange(scores.shape[0])
            acc: torch.Tensor = i.view(-1) == truth
            outputs.update({"accuracy": acc.float().mean().unsqueeze(0)})
        else:
            query_cls = query_cls.cpu().float().numpy()
            if not self.model_config.inner_product:
                query_cls = augment_xq(query_cls)

            scores, examples = self._full_data.get_nearest_examples_batch(
                "mips_cls",
                queries=query_cls,
                k=self.model_config.top_k,
            )

            outputs.update(
                {
                    "scores": np.array(scores),
                    "examples": examples,
                    "query_cls": query_cls,
                }
            )

            if "aid" in batch and "counts" in batch:
                pred = torch.tensor(
                    [[b in a for a in e["aid"]] for e, b in zip(examples, batch["aid"])]
                ).float()
                outputs.update(retriever_metrics(pred, batch["counts"].cpu()))

        return outputs

    def on_validation_start(self) -> None:
        self.eval()
        if self.model_config.test_full_data:
            if self.model_config.dry_run:
                knowledge_base = self.knowledge_base.select(
                    range(self.model_config.batch_size * 10)
                )

            @torch.inference_mode()
            def _encode(batch):
                batch = self.mips_tokenizer(
                    batch["mips_column"],
                    max_length=self.model_config.mips_tok_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                mips_global_attention_mask = torch.zeros_like(
                    batch["input_ids"], device=self.device
                )
                mips_global_attention_mask[:, 0] = 1
                mips_output = self.mips_encoder(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    global_attention_mask=mips_global_attention_mask,
                )
                mips_cls: torch.Tensor = mips_output[0][:, 0, :]

                return {"cls": mips_cls.cpu().float().numpy()}

            self._full_data = knowledge_base.map(
                _encode,
                batched=True,
                batch_size=self.model_config.validation_batch_size,
                load_from_cache_file=False,
                desc="Knowledge Base Encoding...",
            )
            self._full_data.set_format("numpy", ["cls"], True)

            if not self.model_config.inner_product:
                phi = self._full_data.map(
                    lambda x: {"p": (x["cls"] ** 2).sum(1)},
                    batched=True,
                    load_from_cache_file=False,
                    desc="Calculating Phi..",
                )
                phi = max(phi["p"])

                self._full_data = self._full_data.map(
                    lambda x: {"cls": augment_xb(x["cls"], phi)},
                    batched=True,
                    load_from_cache_file=False,
                    desc="Augmenting matrix...",
                )

            metric = (
                faiss.METRIC_INNER_PRODUCT
                if self.model_config.inner_product
                else faiss.METRIC_L2
            )
            self._full_data.add_faiss_index(
                column="cls",
                index_name="mips_cls",
                metric_type=metric,
            )

    def on_predict_start(self) -> None:
        self.on_validation_start()

    def on_test_start(self) -> None:
        self.on_validation_start()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.predict_step(
            batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        return outputs

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = self.predict_step(
            batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )
        return outputs

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.model_config.learning_rate)
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = DeepSpeedCPUAdam(
                self.parameters(), lr=self.model_config.learning_rate
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.model_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=0.01,
            )
        # optimizer = FusedAdam(self.parameters(), lr=self.model_config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.model_config.warmup_steps,
            num_training_steps=self.trainer.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class BOWModel(nn.Module):
    def __init__(self, tgt_embed, tokenizer):
        # bag of words autoencoder
        super(BOWModel, self).__init__()
        vocab_size, embed_dim = tgt_embed.weight.shape
        self.tokenizer = tokenizer
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False,
        )
        self.output_projection.weight = tgt_embed.weight
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, outs, label, attention_mask):
        logits = self.output_projection(self.proj(outs))
        lprobs = F.log_softmax(logits, dim=-1)

        # bsz x vocab
        label_mask = (
            (label == self.tokenizer.unk_token_id)
            | (label == self.tokenizer.cls_token_id)
            | (label == self.tokenizer.eos_token_id)
            | (~attention_mask.bool())
        )
        # label_mask = torch.le(label, 3)  # except for PAD UNK BOS EOS
        loss = torch.gather(-lprobs, -1, label).masked_fill(label_mask, 0.0)
        loss = loss.sum(dim=-1).mean()

        return loss
