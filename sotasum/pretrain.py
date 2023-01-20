import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import logging
import argparse
import rich
import pymsteams

from mlflow.client import MlflowClient
from transformers import get_linear_schedule_with_warmup, LongformerModel, LongformerConfig, LongformerTokenizer
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, RichModelSummary, RichProgressBar
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.loggers import MLFlowLogger
from data_loaders import PretrainMultiXScienceDataset
from torch.utils.data import DataLoader
from rich.traceback import install


def fault_tolerant(func):
    def wrapper(*args, **kwargs):
        try:
            x = func(*args, **kwargs)
        except:
            return None
        return x
    return wrapper


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
        rouge_keys = ["val_accuracy"]
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
            self.query_linear = nn.Linear(
                in_features=query_config.hidden_size,
                out_features=mips_config.vocab_size,
                bias=True,
            )
            self.mips_linear = nn.Linear(
                in_features=mips_config.hidden_size,
                out_features=query_config.vocab_size,
                bias=True,
            )

    @rank_zero_only
    def _log_params(self) -> None:
        params = {k: str(v) for k, v in vars(self.args).items()}
        self.logger.log_hyperparams(params)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._log_params()

    def _get_dataloader(self, mode: str) -> DataLoader:
        data = PretrainMultiXScienceDataset(
            args=self.args,
            mode=mode,
            query_tokenizer=self.query_tokenizer,
            query_tokenizer_kwargs={
                "max_length": self.args.query_tok_max_length,
            },
            mips_tokenizer=self.mips_tokenizer,
            mips_tokenizer_kwargs={
                "max_length": self.args.mips_tok_max_length,
            },
        )
        dataloader = DataLoader(
            dataset=data,
            batch_size=self.args.batch_size,
            num_workers=self.args.data_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(mode="validation")

    def test_dataloader(self) -> DataLoader:
        if self.args.dry_run:
            dataloader = self._get_dataloader(mode="test")
        else:
            dataloader = self._get_dataloader(mode="test")
        return dataloader

    def training_step(self, batch, batch_idx):
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']

        mips_input_ids = batch['mips_input_ids']
        mips_attention_mask = batch['mips_attention_mask']

        query_global_attention_mask = torch.zeros_like(
            query_input_ids, device=self.device)
        query_global_attention_mask[:, 0] = 1
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
            sentence_scores.shape[0]).to(self.device)
        sentence_loss = nn.functional.cross_entropy(
            input=sentence_scores,
            target=sentence_target,
        )

        loss = sentence_loss

        loss_dict = {
            "loss": loss.item(),
        }

        if self.args.token_loss:
            query_token: torch.Tensor = self.query_linear(query_cls)
            mips_token: torch.Tensor = self.mips_linear(mips_cls)

            query_mask = (mips_input_ids == self.mips_tokenizer.unk_token_id) | (mips_input_ids == self.mips_tokenizer.cls_token_id) | (
                mips_input_ids == self.mips_tokenizer.eos_token_id) | (~mips_attention_mask.bool())
            mips_mask = (query_input_ids == self.query_tokenizer.unk_token_id) | (query_input_ids == self.query_tokenizer.cls_token_id) | (
                query_input_ids == self.query_tokenizer.eos_token_id) | (~query_attention_mask.bool())

            query_token_scores = -(query_token.log_softmax(-1))
            mips_token_scores = -(mips_token.log_softmax(-1))

            query_token_loss = query_token_scores.gather(
                -1, mips_input_ids).masked_fill(query_mask, 0.)
            mips_token_loss = mips_token_scores.gather(
                -1, query_input_ids).masked_fill(mips_mask, 0.)

            query_token_loss = query_token_loss.sum(-1).mean()
            mips_token_loss = mips_token_loss.sum(-1).mean()
            token_loss = query_token_loss + mips_token_loss

            loss += token_loss
            loss_dict["token_loss"] = token_loss.item()
            loss_dict["sentence_loss"] = sentence_loss.item()

        self.losses.append(loss_dict)

        with torch.no_grad():
            scores = F.normalize(query_cls) @ F.normalize(mips_cls).T
            _, i = scores.topk(1)
            truth = torch.arange(scores.shape[0]).to(self.device)
            loss_dict['train_accuracy'] = (
                i.view(-1) == truth).float().mean().item()

        return loss

    def validation_step(self, batch, batch_idx):
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']

        mips_input_ids = batch['mips_input_ids']
        mips_attention_mask = batch['mips_attention_mask']

        query_global_attention_mask = torch.zeros_like(
            query_input_ids, device=self.device)
        query_global_attention_mask[:, 0] = 1
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

        scores = query_cls @ mips_cls.T
        _, i = scores.topk(1)

        truth = torch.arange(scores.shape[0]).to(self.device)
        acc = (i.view(-1) == truth).float().mean().unsqueeze(0)

        return acc

    def validation_epoch_end(self, outputs) -> None:
        accuracy = torch.cat(outputs).mean()
        self.log("val_accuracy", accuracy, sync_dist=True)

    def test_step(self, batch, batch_idx) -> None:
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    def test_epoch_end(self, outputs) -> None:
        accuracy = torch.cat(outputs).mean()
        self.log("test_accuracy", accuracy, sync_dist=True)

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
        if len(self.losses) == self.args.accumulate_grad_batches:
            metrics_dict = {k: np.mean([l[k] for l in self.losses])
                            for k in self.losses[0].keys()}
            self.logger.log_metrics(metrics_dict, step=self.global_step)
            self.losses.clear()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.args.lr)
        # optimizer = FusedAdam(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def pretrain(args: argparse.Namespace):
    rich.print(*[f"{k} = {v}" for k, v in vars(args).items()], sep='\n')

    logger = MLFlowLogger(
        experiment_name=f"{args.mlflow_exp_prefix}_pretrain_sotasum",
        tracking_uri=f"file:{args.mlflow_mlruns_dir}/mlruns",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        save_top_k=1,
        monitor="val_accuracy",
        mode="max",
        filename='{epoch}-{val_accuracy:.2f}',
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

    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        # offload_parameters=True,
        logging_level=ds_logging_level,
        initial_scale_power=4,
        # allgather_bucket_size=5e8,
        # reduce_bucket_size=5e8,
    )
    # strategy = None

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
    )

    model = RetrieverLightning(
        args=args,
    )

    trainer.fit(
        model=model,
    )


def test(args: argparse.Namespace):
    ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN

    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        # offload_parameters=True,
        logging_level=ds_logging_level,
        initial_scale_power=4,
        # allgather_bucket_size=5e8,
        # reduce_bucket_size=5e8,
    )

    progress_bar_callback = RichProgressBar(
        refresh_rate=1,
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


def get_args(args: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # Script args
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pb-refresh-rate", type=int, default=1)
    parser.add_argument("--mlflow-exp-prefix", type=str, default="")
    parser.add_argument("--mlflow-mlruns-dir", type=str, default=".")
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

    # Training args
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="models")

    # Validation args
    parser.add_argument("--validation-batch-size", type=int, default=16)

    # Data args
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument("--data-path", type=str, default='../data_hf')

    # Test args
    parser.add_argument("--trained-checkpoint", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--save-models-dir", type=str, default=None)

    return parser.parse_args(args)


if __name__ == "__main__":
    install(show_locals=False)
    args = get_args()

    if args.mode == "train":
        pretrain(args=args)
    elif args.mode == "test":
        test(args=args)
    else:
        raise NotImplementedError(f"{args.mode} is not implemented.")
