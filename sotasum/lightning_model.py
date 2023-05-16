import numpy as np
import json
import pytorch_lightning as pl
import torch
import pymsteams
import rich

from model_config import ModelConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from retriever_generator import RetrieverGenerator
from transformers import get_linear_schedule_with_warmup
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from evaluate import load
from loss import label_smoothed_nll_loss_transformers
from data_loaders import MultiXScienceDataset, MultiXScienceDualDataset, MultiXScienceAggregatedDataset
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

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.args = model_config
        self.losses = []
        self.scores = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.retriever_generator = None
        self.rouge = load(
            self.args.rouge_path,
            experiment_id=self.args.mips_cache_prefix,
        )

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

        self.filename = "output"

        # self.example_input_array = {
        #     'input_ids': torch.ones((2, 5)).long(),
        #     'attention_mask': torch.ones((2, 5)),
        #     'decoder_input_ids': torch.ones((2, 5)).long(),
        #     'decoder_attention_mask': torch.ones((2, 5)),
        #     'query_input_ids': torch.ones((2, 5)).long(),
        #     "query_attention_mask": torch.ones((2, 5)),
        #     "index": None,
        # }

    # @rank_zero_only
    # def _log_params(self) -> None:
    #     for logger in self.loggers:
    #         if isinstance(logger, MLFlowLogger):
    #             params = {k: str(v) for k, v in vars(self.args).items()}
    #             logger.log_hyperparams(params)

    def setup(self, stage: str) -> None:
        if self.retriever_generator == None:
            self.retriever_generator = RetrieverGenerator(args=self.args)

        # for logger in self.loggers:
        #     if isinstance(logger, MLFlowLogger):
        #         mlflow_run = logger.experiment.get_run(logger.run_id)

        if stage == "fit":
            # self._log_params()
            if self.args.mips_freezed and not self.args.mips_disabled:
                self.retriever_generator.encoder.mips.encoder.requires_grad_(
                    False)
                self.retriever_generator.encoder.query_encoder.requires_grad_(
                    False)
            elif self.args.mips_encoder_freezed and not self.args.mips_disabled:
                self.retriever_generator.encoder.mips.encoder.requires_grad_(
                    False)

        if not self.args.mips_no_init_build and not self.args.mips_disabled:
            self.retriever_generator.encoder._build_mips_index(self.local_rank)

    def on_train_batch_start(self, batch: dict, batch_idx: int, unused: int = 0) -> None:
        if not self.args.mips_no_init_build and not self.args.mips_disabled and not self.args.mips_freezed and not self.args.mips_encoder_freezed:
            self.retriever_generator.encoder._update_mips_index(
                global_step=self.global_step,
                local_rank=self.local_rank,
            )

    def _get_data_loader(self, mode: str, batch_size: int, select_indices: list = None) -> DataLoader:
        if self.args.dataset_name == "multi_x_science":
            if self.args.decoder_max_length is None:
                decoder_max_length = self.retriever_generator.config.max_decoder_position_embeddings
            else:
                decoder_max_length = self.args.decoder_max_length
            if self.args.multi_x_science_dataset_mode == "dual":
                data = MultiXScienceDualDataset(
                    args=self.args,
                    mode=mode,
                    tokenizer=self.retriever_generator.tokenizer,
                    tokenizer_kwargs=self.tokenizer_kwargs,
                    query_tokenizer=self.retriever_generator.encoder.query_tokenizer,
                    query_tokenizer_kwargs=self.query_tokenizer_kwargs,
                    select_indices=select_indices,
                    decoder_max_length=decoder_max_length,
                )
            elif self.args.multi_x_science_dataset_mode == "original":
                data = MultiXScienceDataset(
                    args=self.args,
                    mode=mode,
                    tokenizer=self.retriever_generator.tokenizer,
                    tokenizer_kwargs=self.tokenizer_kwargs,
                    query_tokenizer=self.retriever_generator.encoder.query_tokenizer,
                    query_tokenizer_kwargs=self.query_tokenizer_kwargs,
                    select_indices=select_indices,
                    decoder_max_length=decoder_max_length,
                )
            elif self.args.multi_x_science_dataset_mode == "aggregated":
                data = MultiXScienceAggregatedDataset(
                    args=self.args,
                    mode=mode,
                    tokenizer=self.retriever_generator.tokenizer,
                    tokenizer_kwargs=self.tokenizer_kwargs,
                    query_tokenizer=self.retriever_generator.encoder.query_tokenizer,
                    query_tokenizer_kwargs=self.query_tokenizer_kwargs,
                    select_indices=select_indices,
                    decoder_max_length=decoder_max_length,
                )
        else:
            assert False, "Unknown dataset name, please choose from these names : multi_x_science"
        data_loader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            num_workers=self.args.data_workers,
            shuffle=False,
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
        aid: list,
        aid_counts: torch.Tensor,
        target_str: list,
        input_str: list = None,
        **kwargs
    ):

        indices = None if self.args.memory_forcing == "retrieved_forcing" else index

        encoder_outputs = self.retriever_generator.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            mips_ignore_indexes=indices,
            aid=aid,
            aid_counts=aid_counts,
            target_str=target_str,
            input_str=input_str,
            return_dict=True,
            logger=self.log_dict,
        )

        # decoder_input_ids_shifted = decoder_input_ids[:, :-1]
        # target_shifted = decoder_input_ids[:, 1:].clone()

        decoder_input_ids_shifted = self.retriever_generator.model.prepare_decoder_input_ids_from_labels(
            decoder_input_ids)
        target_shifted = decoder_input_ids.clone()

        decoder_head_outputs = self.retriever_generator(
            input_ids=decoder_input_ids_shifted,
            attention_mask=None,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        if not self.args.mips_disabled:
            lprobs = decoder_head_outputs.logits
        else:
            lprobs = torch.nn.functional.log_softmax(
                decoder_head_outputs.logits, dim=-1)

        if self.args.log_copy_metrics:
            k = 10
            copy_gate: torch.Tensor = decoder_head_outputs.copy_gate
            copy_probs: torch.Tensor = decoder_head_outputs.copy_probs

            memory_length = copy_probs.shape[2] // self.args.mips_topk

            all_max_copy_probs, all_index = copy_probs.max(2)
            topk_max_copy_probs, topk_index = all_max_copy_probs.topk(k)

            topk_index = torch.div(all_index.gather(
                1, topk_index), memory_length, rounding_mode="floor")
            all_index = torch.div(
                all_index, memory_length, rounding_mode="floor")

            self.log_dict(
                {
                    "copy_gate_max_mean": copy_gate.max(1)[0].mean().item(),
                    "copy_gate_mean": copy_gate.mean().item(),
                    "copy_probs_max_mean": all_max_copy_probs.mean().item(),
                    # Moyenne des probas max sur tous les tokens.
                    f"copy_probs_top{k}_mean": topk_max_copy_probs.mean().item(),
                    # Moyenne des probas max sur les k tokens ayant la plus grande proba.
                    "all_index": all_index.float().mean().item(),
                    f"top{k}_index": topk_index.float().mean().item(),
                },
                sync_dist=True,
            )

        return lprobs, target_shifted

    def training_step(self, batch, batch_idx):
        self.train()
        if not self.args.mips_disabled:
            self.retriever_generator.encoder.query_encoder.train(
                not self.args.mips_freezed)
            self.retriever_generator.encoder.mips.encoder.train(
                not (self.args.mips_freezed or self.args.mips_encoder_freezed))

        lprobs, target_shifted = self(**batch)

        loss = label_smoothed_nll_loss_transformers(
            log_probs=-lprobs,
            labels=target_shifted,
            epsilon=self.args.label_smoothing_eps,
            ignore_index=self.retriever_generator.pad_token_id,
        )

        self.losses.append(loss.item())
        if len(self.losses) == self.trainer.accumulate_grad_batches:
            self.log('train_loss', np.mean(self.losses), prog_bar=False,
                     on_step=True, sync_dist=True, on_epoch=False)
            self.losses.clear()

        return loss

    @torch.inference_mode()
    def generate(self, batch: dict):
        self.eval()
        kwargs = {
            "attention_mask": batch["attention_mask"],
            "query_input_ids": batch['query_input_ids'],
            "query_attention_mask": batch["query_attention_mask"],
            "input_str": batch.get('input_str', None)
        }

        output = self.retriever_generator.generate(
            inputs=batch['input_ids'],
            max_length=self.args.generate_max_length,
            min_length=self.args.generate_min_length,
            no_repeat_ngram_size=self.args.generate_no_repeat_ngram_size,
            length_penalty=self.args.generate_length_penalty,
            num_beams=self.args.num_beams,
            do_sample=False,
            use_cache=self.args.use_cache,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

        return output

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        self.retriever_generator.copy_probs = tuple()
        output = self.generate(batch)

        predictions = self.retriever_generator.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True)
        references = self.retriever_generator.tokenizer.batch_decode(
            batch['decoder_input_ids'], skip_special_tokens=True)

        current_examples = None
        if hasattr(self.retriever_generator.encoder, "mips"):
            current_examples = self.retriever_generator.encoder.mips.examples

        tokens, tokens_copy_probs = None, None
        if self.args.output_copy_probs and self.retriever_generator.copy_probs != None:
            copy_probs = torch.cat(self.retriever_generator.copy_probs, 1)
            if hasattr(output, "beam_indices"):
                copy_probs = torch.stack([copy_probs[output.beam_indices[:, i], i, :] for i in range(
                    output.beam_indices.shape[1]-1)]).transpose(0, 1)

            tokens_copy_probs = copy_probs.gather(
                2, output.sequences[:, :-1].unsqueeze(-1)).squeeze().tolist()
            tokens = [self.retriever_generator.tokenizer.convert_ids_to_tokens(
                seq) for seq in output.sequences]

        output = {
            "predictions": predictions,
            "references": references,
            "examples": current_examples,
            "tokens": tokens,
            "tokens_copy_probs": tokens_copy_probs,
        }

        return output

    def val_dataloader(self) -> DataLoader:
        return self._get_data_loader("validation", batch_size=self.args.validation_batch_size)

    def validation_step(self, batch, batch_idx) -> dict:
        output = self.predict_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        self.rouge.add_batch(
            predictions=output['predictions'],
            references=output['references'],
        )
        return output

    def on_validation_epoch_end(self) -> None:
        rouge_scores = self.rouge.compute()
        self.scores.append(rouge_scores)
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)

        with open(f"./{self.filename}.json", mode="w") as f:
            json.dump(self.validation_step_outputs, f)

        self.validation_step_outputs.clear()

    def test_dataloader(self) -> DataLoader:
        return self._get_data_loader("test", batch_size=self.args.batch_size)

    def test_step(self, batch, batch_idx) -> dict:
        output = self.predict_step(batch, batch_idx)
        self.test_step_outputs.append(output)
        self.rouge.add_batch(
            predictions=output['predictions'],
            references=output['references'],
        )
        return output

    def on_test_epoch_end(self) -> tuple:
        rouge_scores = self.rouge.compute()
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)

        splited_path = self.args.checkpoint_path.split("/")
        ckpt_name = splited_path[-1].replace('.ckpt', "")
        ckpt_type = splited_path[-2]
        args = f"num_beams={self.args.num_beams}"
        with open(f"./{ckpt_type}-{ckpt_name}-{args}-outputs.json", mode="w") as f:
            json.dump(self.test_step_outputs, f)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(), lr=self.args.lr)
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        # optimizer = FusedAdam(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
