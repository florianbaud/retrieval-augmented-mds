import os
import numpy as np
import json
import pytorch_lightning as pl
import argparse
import torch
import pymsteams
import rich

from rich.console import Console
from rich.text import Text
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from retriever_generator import RetrieverGenerator
from random import random
from transformers import get_linear_schedule_with_warmup
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from evaluate import load
from loss import label_smoothed_nll_loss_copy, label_smoothed_nll_loss_custom, label_smoothed_nll_loss_fairseq
from data_loaders import MultiXScienceDataset
from torch.utils.data import DataLoader


def fault_tolerant(func):
    def wrapper(*args, **kwargs):
        try:
            x = func(*args, **kwargs)
        except:
            return None
        return x
    return wrapper


class GradientsPrintingCallback(Callback):

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for n, p in pl_module.named_parameters():
            pl_module.print(n, ":", p.grad)


class TeamsCallback(Callback):

    def __init__(self, hookurl: str) -> None:
        super().__init__()
        self.hookurl = hookurl

    @fault_tolerant
    @rank_zero_only
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title("🚀 Training started")
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
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title("✔️ Validation End")

        rouge_scores = pl_module.scores[-1]
        if isinstance(rouge_scores, dict):
            section = pymsteams.cardsection()
            section.title("Rouge Scores")
            for k, v in rouge_scores.items():
                section.addFact(k, v)
            msg.addSection(section)

        msg.text('Check out validation summary :')
        msg.send()


class LongformerLightning(pl.LightningModule):

    def __init__(self, args: argparse.Namespace, model: RetrieverGenerator) -> None:
        super().__init__()
        self.args = args
        self.losses = []
        self.scores = []

        self.retriever_generator = model
        self.rouge = load(self.args.rouge_path)

        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
            "max_length": self.args.model_tok_max_length
        }

        self.query_tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
            "max_length": self.args.query_tok_max_length
        }

        # self.example_input_array = {
        #     'input_ids': torch.ones((2, 5)).long(),
        #     'attention_mask': torch.ones((2, 5)),
        #     'decoder_input_ids': torch.ones((2, 5)).long(),
        #     'decoder_attention_mask': torch.ones((2, 5)),
        #     'query_input_ids': torch.ones((2, 5)).long(),
        #     "query_attention_mask": torch.ones((2, 5)),
        #     "index": None,
        # }

    @rank_zero_only
    def _log_params(self) -> None:
        for logger in self.loggers:
            if isinstance(logger, MLFlowLogger):
                params = {k: str(v) for k, v in vars(self.args).items()}
                logger.log_hyperparams(params)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._log_params()
            if self.args.mips_freezed and not self.args.mips_disabled:
                self.retriever_generator.encoder.mips.encoder.requires_grad_(
                    False)

    def on_fit_start(self) -> None:
        if not self.args.mips_no_init_build and not self.args.mips_disabled:
            self.retriever_generator.encoder._build_mips_index(self.local_rank)

    def on_predict_start(self) -> None:
        self.on_fit_start()

    def on_test_start(self) -> None:
        self.on_fit_start()

    def on_train_batch_start(self, batch: dict, batch_idx: int, unused: int = 0) -> None:
        if not self.args.mips_no_init_build and not self.args.mips_disabled and not self.args.mips_freezed:
            self.retriever_generator.encoder._update_mips_index(
                global_step=self.global_step,
                local_rank=self.local_rank,
            )

    def _get_data_loader(self, mode: str, batch_size: int, select_indices: list = None) -> DataLoader:
        if self.args.dataset_name == "multi_x_science":
            data = MultiXScienceDataset(
                args=self.args,
                mode=mode,
                model_config=self.retriever_generator.config,
                tokenizer=self.retriever_generator.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                query_tokenizer=self.retriever_generator.encoder.query_tokenizer,
                query_tokenizer_kwargs=self.query_tokenizer_kwargs,
                select_indices=select_indices,
            )
        else:
            assert False, "Unknown dataset name, please choose from these names : multi_x_science"
        data_loader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            num_workers=self.args.data_workers,
        )
        return data_loader

    def train_dataloader(self) -> DataLoader:
        return self._get_data_loader("train", batch_size=self.args.batch_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        index: list,
        target_str: list,
        **kwargs
    ):

        # indexes = None if self.args.copy_forcing > random() else index

        encoder_outputs = self.retriever_generator.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            mips_ignore_indexes=index,
            target_str=target_str,
            return_dict=True,
        )

        decoder_input_ids_shifted = decoder_input_ids[:, :-1]
        target_shifted = decoder_input_ids[:, 1:].clone()

        decoder_head_outputs = self.retriever_generator(
            input_ids=decoder_input_ids_shifted,
            attention_mask=None,
            encoder_outputs=encoder_outputs,
            use_cache=False,
            return_dict=True,
        )

        if not self.args.mips_disabled:
            lprobs = decoder_head_outputs.logits
        else:
            lprobs = torch.nn.functional.log_softmax(
                decoder_head_outputs.logits, dim=-1)

        return lprobs, target_shifted

    def training_step(self, batch, batch_idx):

        lprobs, target_shifted = self(**batch)

        loss, _ = label_smoothed_nll_loss_custom(
            lprobs=lprobs,
            target=target_shifted,
            epsilon=self.args.label_smoothing_eps,
            ignore_index=self.retriever_generator.pad_token_id,
            reduce="mean",
        )

        self.losses.append(loss.item())
        if len(self.losses) == self.args.accumulate_grad_batches:
            self.log('train_loss', np.mean(self.losses), prog_bar=False,
                     on_step=True, sync_dist=True, on_epoch=False)
            self.losses.clear()

        return loss

    def generate(self, batch: dict):
        kwargs = {
            "query_input_ids": batch['query_input_ids'],
            "query_attention_mask": batch["query_attention_mask"],
        }

        output = self.retriever_generator.generate(
            inputs=batch['input_ids'],
            max_length=self.args.generate_max_length,
            num_beams=self.args.num_beams,
            do_sample=False,
            use_cache=self.args.use_cache,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

        return output

    def val_dataloader(self) -> DataLoader:
        return self._get_data_loader("validation", batch_size=self.args.validation_batch_size)

    def validation_step(self, batch, batch_idx) -> tuple:
        output = self.generate(batch)
        predictions = self.retriever_generator.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True)
        references = self.retriever_generator.tokenizer.batch_decode(
            batch['decoder_input_ids'], skip_special_tokens=True)
        self.rouge.add_batch(predictions=predictions, references=references)
        return predictions, references

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.eval()
        console = Console(record=True)
        self.retriever_generator.copy_probs = tuple()
        output = self.generate(batch)
        copy_probs = torch.cat(self.retriever_generator.copy_probs, 1)

        # rich.print(copy_probs.sum())
        # rich.print(output.beam_indices)
        # rich.print(output.sequences)
        # rich.print(copy_probs.shape)
        # rich.print(output.beam_indices.shape)
        # rich.print(output.sequences.shape)
        # exit()

        beam_copy_probs = torch.stack(
            [copy_probs[output.beam_indices[:, i], i, :]
                for i in range(output.beam_indices.shape[1]-1)]
        ).transpose(0, 1)
        tokens_copy_probs = beam_copy_probs.gather(
            2, output.sequences[:, :-1].unsqueeze(-1)).squeeze()

        current_examples = self.retriever_generator.encoder.mips.examples

        predictions = self.retriever_generator.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True)
        references = self.retriever_generator.tokenizer.batch_decode(
            batch['decoder_input_ids'], skip_special_tokens=True)

        colors = ["#33FF00", "#33FF33", "#33FF66",
                  "#33FF99", "#33FFCC", "#33FFFF"]

        texts = []
        for seq, copy_probs in zip(output.sequences, tokens_copy_probs):
            tokens = self.retriever_generator.tokenizer.convert_ids_to_tokens(
                seq)
            text = Text()
            for token, prob in zip(tokens, copy_probs):
                color_id = int(prob // (1/len(colors)))
                text.append(f"{token}-{prob} ", style=colors[color_id])
            texts.append(text)

        for t in texts:
            console.print(t)

        # console.save_html(f"test{self.local_rank}.html")
        # console.save_svg(f"test{self.local_rank}.svg")

        return predictions, references, current_examples

    def validation_epoch_end(self, outputs) -> None:
        dict_outputs = [
            {"predictions": p, "target": r} for o in outputs for p, r in zip(o[0], o[1])
        ]
        outputs_path = os.path.join(
            self.args.validation_outputs_dir,
            f'validation_outputs_{self.current_epoch}.json',
        )
        with open(outputs_path, mode='w', encoding="utf-8") as f:
            json.dump(dict_outputs, f)
        rouge_scores = self.rouge.compute()
        self.scores.append(rouge_scores)
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)
        return rouge_scores

    def test_step(self, batch, batch_idx) -> tuple:
        self.retriever_generator.copy_probs = tuple()

        output = self.generate(batch)
        copy_probs = torch.cat(self.retriever_generator.copy_probs, 1)

        predictions = self.retriever_generator.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True)
        references = self.retriever_generator.tokenizer.batch_decode(
            batch['decoder_input_ids'], skip_special_tokens=True)
        self.rouge.add_batch(predictions=predictions, references=references)
        return predictions, references

    def test_dataloader(self) -> DataLoader:
        return self._get_data_loader("test", batch_size=self.args.batch_size)

    def test_epoch_end(self, outputs) -> None:
        dict_outputs = [
            {"predictions": p, "target": r} for o in outputs for p, r in zip(o[0], o[1])
        ]
        # outputs_path = os.path.join(
        #     self.args.validation_outputs_dir,
        #     f'validation_outputs_{self.current_epoch}.json',
        # )
        # with open(outputs_path, mode='w', encoding="utf-8") as f:
        #     json.dump(dict_outputs, f)
        rouge_scores = self.rouge.compute()
        self.scores.append(rouge_scores)
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)

        for i, t in enumerate(dict_outputs):
            textmd = f"""{t['predictions']}\n\n---\n\n{t['target']}"""
            self.logger.experiment.add_text(
                'outputs',
                textmd,
                global_step=i,
            )
        return rouge_scores

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
