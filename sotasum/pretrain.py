import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import logging
import argparse
import rich
import faiss
import pymsteams

from datasets import disable_caching
from mlflow.client import MlflowClient
from transformers import get_linear_schedule_with_warmup, LongformerModel, LongformerConfig, LongformerTokenizer
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, RichModelSummary, RichProgressBar
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from data_loaders import PretrainMultiXScienceDataset, PretrainAbstractMultiXScienceDataset, load_mips_multi_x_science
from torch.utils.data import DataLoader
from rich.traceback import install


def get_phi(xb: np.ndarray):
    return (xb ** 2).sum(1).max()


def augment_xb(xb, phi=None):
    norms = (xb ** 2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((xb, extracol.reshape(-1, 1)))


def augment_xq(xq):
    extracol = np.zeros(len(xq), dtype='float32')
    return np.hstack((xq, extracol.reshape(-1, 1)))


def fault_tolerant(func):
    def wrapper(*args, **kwargs):
        try:
            x = func(*args, **kwargs)
        except:
            return None
        return x
    return wrapper


def retriever_metrics(pred: torch.Tensor, counts: torch.Tensor) -> dict:
    recall = (pred.sum(-1) / counts).mean().item()

    reciprocal_rank = (1 / pred.argmax(-1))
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


class TeamsCallback(Callback):

    def __init__(self, hookurl: str) -> None:
        super().__init__()
        self.hookurl = hookurl

    @fault_tolerant
    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title("🚀 Pretraining started")
        msg.text("Fit loop begins.")
        msg.send()

    @fault_tolerant
    @rank_zero_only
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title("❌ Training exception")
        msg.text(f'An error occured : \n{exception}')
        msg.send()

    @fault_tolerant
    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mlflow_client: MlflowClient = pl_module.logger.experiment
        rouge_keys = ["recall", "reciprocal_rank", "average_precision"]
        rouge_scores = {
            k: getattr(mlflow_client.get_metric_history(pl_module.logger.run_id, k)[-1], 'value', None) for k in rouge_keys
        }

        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title("✔️ Validation End")
        section = pymsteams.cardsection()
        section.title("Validation accuracy")

        for k, v in rouge_scores.items():
            section.addFact(k, v)

        msg.addSection(section)
        msg.text('Check out validation summary :')
        msg.send()


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
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, outs, label, attention_mask):
        logits = self.output_projection(self.proj(outs))
        lprobs = F.log_softmax(logits, dim=-1)

        # bsz x vocab
        label_mask = (label == self.tokenizer.unk_token_id) | (label == self.tokenizer.cls_token_id) | (
            label == self.tokenizer.eos_token_id) | (~attention_mask.bool())
        # label_mask = torch.le(label, 3)  # except for PAD UNK BOS EOS
        loss = torch.gather(-lprobs, -1, label).masked_fill(label_mask, 0.)
        loss = loss.sum(dim=-1).mean()

        return loss


class RetrieverLightning(pl.LightningModule):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.losses = []

        self.query_tokenizer = LongformerTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.query_encoder_path,
        )
        query_config = LongformerConfig.from_pretrained(
            pretrained_model_name_or_path=self.args.query_encoder_path,
            gradient_checkpointing=self.args.gradient_checkpointing,
        )
        self.query_encoder = LongformerModel.from_pretrained(
            pretrained_model_name_or_path=self.args.query_encoder_path,
            config=query_config,
        )

        # _ = self.query_tokenizer.add_special_tokens(
        #     {"additional_special_tokens": [self.args.doc_sep]})
        # _ = self.query_encoder.resize_token_embeddings(
        #     len(self.query_tokenizer))
        # self.doc_sep_id = self.query_tokenizer.convert_tokens_to_ids(
        #     self.args.doc_sep)

        if self.args.query_state_dict is not None:
            query_state_dict = torch.load(self.args.query_state_dict)
            self.query_encoder.load_state_dict(query_state_dict)

        self.mips_tokenizer = LongformerTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.model_name,
        )
        mips_config = LongformerConfig.from_pretrained(
            pretrained_model_name_or_path=self.args.model_name,
            gradient_checkpointing=self.args.gradient_checkpointing,
        )
        self.mips_encoder = LongformerModel.from_pretrained(
            pretrained_model_name_or_path=self.args.model_name,
            config=mips_config,
        )

        if self.args.mips_state_dict is not None:
            mips_state_dict = torch.load(self.args.mips_state_dict)
            self.mips_encoder.load_state_dict(mips_state_dict)

        if self.args.pooling:
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

        if self.args.token_loss:
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

    @rank_zero_only
    def _log_params(self) -> None:
        self.save_hyperparameters(self.args)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._log_params()

    def _get_dataloader(self, mode: str) -> DataLoader:
        query_tokenizer_kwargs = {
            "max_length": self.args.query_tok_max_length,
        }
        mips_tokenizer_kwargs = {
            "max_length": self.args.mips_tok_max_length,
        }
        if self.args.pretrain_dataset == "multi_x_science_abstract":
            data = PretrainAbstractMultiXScienceDataset(
                args=self.args,
                mode=mode,
                query_tokenizer=self.query_tokenizer,
                query_tokenizer_kwargs=query_tokenizer_kwargs,
                mips_tokenizer=self.mips_tokenizer,
                mips_tokenizer_kwargs=mips_tokenizer_kwargs,
            )
        elif self.args.pretrain_dataset == "multi_x_science_related_work":
            data = PretrainMultiXScienceDataset(
                args=self.args,
                mode=mode,
                query_tokenizer=self.query_tokenizer,
                query_tokenizer_kwargs=query_tokenizer_kwargs,
                mips_tokenizer=self.mips_tokenizer,
                mips_tokenizer_kwargs=mips_tokenizer_kwargs,
            )
        dataloader = DataLoader(
            dataset=data,
            batch_size=self.args.batch_size,
            num_workers=self.args.data_workers,
            shuffle=mode == "train",
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(mode="validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(mode="test")

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(mode="train")

    def training_step(self, batch, batch_idx):
        self.train()
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']

        mips_input_ids = batch['mips_input_ids']
        mips_attention_mask = batch['mips_attention_mask']

        query_global_attention_mask = torch.zeros_like(
            query_input_ids, device=self.device)
        query_global_attention_mask[:, 0] = 1
        # doc_token = query_input_ids == self.doc_sep_id
        # query_global_attention_mask[doc_token] = 1
        query_output = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            global_attention_mask=query_global_attention_mask,
        )

        mips_global_attention_mask = torch.zeros_like(
            mips_input_ids, device=self.device)
        mips_global_attention_mask[:, 0] = 1
        mips_output = self.mips_encoder(
            input_ids=mips_input_ids,
            attention_mask=mips_attention_mask,
            global_attention_mask=mips_global_attention_mask,
        )

        query_cls: torch.Tensor = query_output[0][:, 0, :]
        mips_cls: torch.Tensor = mips_output[0][:, 0, :]

        if self.args.pooling:
            query_cls = self.query_pooling(query_cls)
            mips_cls = self.mips_pooling(mips_cls)

        # query_cls = nn.functional.normalize(
        #     query_cls,
        # )
        # mips_cls = nn.functional.normalize(
        #     mips_cls,
        # )

        sentence_scores = (query_cls @ mips_cls.T) / self.args.temperature
        sentence_target = torch.arange(
            sentence_scores.shape[0], device=self.device)
        sentence_loss = nn.functional.cross_entropy(
            input=sentence_scores,
            target=sentence_target,
        )

        loss = sentence_loss

        loss_dict = {
            "loss": loss.item(),
        }

        if self.args.token_loss:
            query_loss = self.query_bow(
                mips_cls, query_input_ids, query_attention_mask)
            mips_loss = self.mips_bow(
                query_cls, mips_input_ids, mips_attention_mask)
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

            loss_dict["sentence_loss"] = loss_dict['loss']
            token_loss = query_loss + mips_loss
            loss_dict["token_loss"] = token_loss.item()
            loss += token_loss
            loss_dict['loss'] = loss.item()

        with torch.no_grad():
            scores = F.normalize(query_cls) @ F.normalize(mips_cls).T
            _, i = scores.topk(1)
            truth = torch.arange(scores.shape[0]).to(self.device)
            loss_dict['train_accuracy'] = (
                i.view(-1) == truth).float().mean().item()

        self.log_dict(loss_dict, sync_dist=True)

        return loss

    def on_validation_start(self) -> None:
        self.eval()
        if self.args.test_full_data:
            data = load_mips_multi_x_science(
                data_path=self.args.data_path,
                script_path="multi_x_science_sum",
                column="ref_abstract" if self.args.pretrain_dataset == "multi_x_science_abstract" else "related_work",
            )

            if self.args.dry_run:
                data = data.select(range(self.args.batch_size*10))

            @torch.inference_mode()
            def _encode(batch):
                batch = self.mips_tokenizer(
                    batch['mips_column'],
                    max_length=self.args.mips_tok_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                mips_global_attention_mask = torch.zeros_like(
                    batch['input_ids'], device=self.device)
                mips_global_attention_mask[:, 0] = 1
                mips_output = self.mips_encoder(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    global_attention_mask=mips_global_attention_mask,
                )
                mips_cls: torch.Tensor = mips_output[0][:, 0, :]

                return {"cls": mips_cls.cpu().float().numpy()}

            self._full_data = data.map(
                _encode,
                batched=True,
                batch_size=self.args.validation_batch_size,
                load_from_cache_file=False,
                desc="Building matrix...",
            )
            self._full_data.set_format('numpy', ['cls'], True)

            if not self.args.inner_product:
                phi = self._full_data.map(
                    lambda x: {"p": (x['cls']**2).sum(1)},
                    batched=True,
                    load_from_cache_file=False,
                    desc="Calculating Phi..",
                )
                phi = max(phi['p'])

                self._full_data = self._full_data.map(
                    lambda x: {"cls": augment_xb(x['cls'], phi)},
                    batched=True,
                    load_from_cache_file=False,
                    desc="Augmenting matrix...",
                )

            metric = faiss.METRIC_INNER_PRODUCT if self.args.inner_product else faiss.METRIC_L2
            self._full_data.add_faiss_index(
                column='cls',
                index_name="mips_cls",
                metric_type=metric,
            )

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        query_global_attention_mask = torch.zeros_like(
            batch['query_input_ids'], device=self.device)
        query_global_attention_mask[:, 0] = 1
        query_output = self.query_encoder(
            input_ids=batch['query_input_ids'],
            attention_mask=batch['query_attention_mask'],
            global_attention_mask=query_global_attention_mask,
        )
        query_cls: torch.Tensor = query_output[0][:, 0, :]

        if getattr(self, "_full_data", None) == None:
            mips_global_attention_mask = torch.zeros_like(
                batch['mips_input_ids'], device=self.device)
            mips_global_attention_mask[:, 0] = 1
            mips_output = self.mips_encoder(
                input_ids=batch['mips_input_ids'],
                attention_mask=batch['mips_attention_mask'],
                global_attention_mask=mips_global_attention_mask,
            )
            mips_cls: torch.Tensor = mips_output[0][:, 0, :]

            scores = query_cls @ mips_cls.T
            _, i = scores.topk(1)

            truth = torch.arange(scores.shape[0])
            acc: torch.Tensor = (i.view(-1) == truth)
            outputs = {
                "accuracy": acc.float().mean().unsqueeze(0),
            }
        else:
            query_cls = query_cls.cpu().float().numpy()
            if not self.args.inner_product:
                query_cls = augment_xq(query_cls)

            scores, examples = self._full_data.get_nearest_examples_batch(
                "mips_cls",
                queries=query_cls,
                k=self.args.top_k,
            )
            pred = torch.tensor([[b in a for a in e['aid']]
                                for e, b in zip(examples, batch['aid'])]).float()
            outputs = retriever_metrics(pred, batch['counts'].cpu())

        return outputs

    def validation_epoch_end(self, outputs) -> None:
        outputs = pd.DataFrame(outputs)
        self.log_dict(outputs.mean().to_dict(), sync_dist=True)

    def on_test_start(self) -> None:
        self.on_validation_start()

    def test_step(self, batch, batch_idx) -> None:
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs) -> None:
        outputs = pd.DataFrame(outputs)
        self.log_dict(outputs.mean().to_dict(), sync_dist=True)

    def on_predict_start(self) -> None:
        return self.on_validation_start()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        rich.print(batch)
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        if self.args.deepspeed:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        # optimizer = FusedAdam(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def pretrain(args: argparse.Namespace):
    rich.print(*[f"{k} = {v}" for k, v in vars(args).items()], sep='\n')

    tb_logger = TensorBoardLogger(
        save_dir=args.tensorboard_save_dir,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=f"{args.mlflow_exp_prefix}_pretrain_sotasum",
        tracking_uri=f"file:{args.mlflow_mlruns_dir}/mlruns",
    )

    logger = [tb_logger, mlflow_logger]

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        save_top_k=1,
        monitor="average_precision",
        mode="max",
        filename='{epoch}-{average_precision:.2f}',
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=args.pb_refresh_rate * args.accumulate_grad_batches,
    )

    profiler = None
    callbacks = [
        checkpoint_callback,
        progress_bar_callback,
    ]

    if args.teams_hookurl is not None:
        callbacks.append(TeamsCallback(args.teams_hookurl))

    ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN

    strategy = None
    if args.deepspeed:
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            # offload_parameters=True,
            logging_level=ds_logging_level,
            initial_scale_power=4,
            # allgather_bucket_size=5e8,
            # reduce_bucket_size=5e8,
        )

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        log_every_n_steps=4,
    )

    model = RetrieverLightning(
        args=args,
    )

    trainer.fit(
        model=model,
    )


def test(args: argparse.Namespace):
    ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN

    strategy = None
    if args.deepspeed:
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            # offload_parameters=True,
            logging_level=ds_logging_level,
            initial_scale_power=4,
            # allgather_bucket_size=5e8,
            # reduce_bucket_size=5e8,
        )

    # progress_bar_callback = RichProgressBar(
    #     refresh_rate=1,
    # )
    progress_bar_callback = TQDMProgressBar(
        refresh_rate=1
    )

    callbacks = [progress_bar_callback]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        strategy=strategy,
    )

    model = RetrieverLightning(
        args=args,
    )

    trainer.test(
        model=model,
        ckpt_path=args.trained_checkpoint,
    )

    if args.save_models_dir is not None:
        mips_save_dir = f"{args.save_models_dir}/mips-{args.model_name.split('/')[-1]}"
        query_save_dir = f"{args.save_models_dir}/query-{args.query_encoder_path.split('/')[-1]}"

        model.mips_encoder.save_pretrained(mips_save_dir)
        model.mips_tokenizer.save_pretrained(mips_save_dir)

        model.query_encoder.save_pretrained(query_save_dir)
        model.query_tokenizer.save_pretrained(query_save_dir)


def predict(args: argparse.Namespace):
    ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN

    # strategy = DeepSpeedStrategy(
    #     stage=2,
    #     offload_optimizer=True,
    #     # offload_parameters=True,
    #     logging_level=ds_logging_level,
    #     initial_scale_power=4,
    #     # allgather_bucket_size=5e8,
    #     # reduce_bucket_size=5e8,
    # )

    # progress_bar_callback = RichProgressBar(
    #     refresh_rate=1,
    # )
    progress_bar_callback = TQDMProgressBar(
        refresh_rate=1
    )

    callbacks = [progress_bar_callback]

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        # strategy=strategy,
    )

    model = RetrieverLightning(
        args=args,
    )

    output = trainer.predict(
        model=model,
    )

    rich.print(output)


def get_args(args: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # Script args
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pb-refresh-rate", type=int, default=1)
    parser.add_argument("--mlflow-exp-prefix", type=str, default="")
    parser.add_argument("--tensorboard-save-dir",
                        type=str, default="pretrain_logs")
    parser.add_argument("--mlflow-mlruns-dir", type=str, default=".")
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--deepspeed-log", action="store_true", default=False)
    parser.add_argument("--teams-hookurl", type=str, default=None)

    # Models args
    parser.add_argument("--model-name", type=str, help="Name of pretrained model.",
                        default="allenai/longformer-large-4096")
    parser.add_argument("--mips-state-dict", type=str, default=None)
    parser.add_argument("--mips-tok-max-length", type=int, default=None)
    parser.add_argument("--query-encoder-path", type=str,
                        default="allenai/longformer-large-4096")
    parser.add_argument("--query-state-dict", type=str, default=None)
    parser.add_argument("--query-tok-max-length", type=int, default=None)
    parser.add_argument("--gradient-checkpointing",
                        action="store_true", default=False)
    parser.add_argument("--token-loss", action="store_true", default=False)
    parser.add_argument("--pooling", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--inner-product", action="store_true", default=False)

    # Training args
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="models")

    # Data args
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument("--data-path", type=str, default='../data_hf')
    parser.add_argument("--doc-sep", type=str, default="<DOC_SEP>")
    parser.add_argument("--pretrain-dataset", choices=[
                        "multi_x_science_abstract", "multi_x_science_related_work"], default="multi_x_science_abstract")

    # Validation / Test args
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--trained-checkpoint", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--save-models-dir", type=str, default=None)
    parser.add_argument("--test-full-data", action="store_true", default=False)

    return parser.parse_args(args)


if __name__ == "__main__":
    install(show_locals=False)
    disable_caching()
    args = get_args()

    if args.mode == "train":
        pretrain(args=args)
    elif args.mode == "test":
        test(args=args)
    elif args.mode == "predict":
        predict(args=args)
    else:
        raise NotImplementedError(f"{args.mode} is not implemented.")
