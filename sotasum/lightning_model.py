import json
import pytorch_lightning as pl
import torch
import pymsteams
import rich

from .model_config import ModelConfig
from .retriever_generator import RetrieverGenerator, RGEncoderModelOutput
from .loss import label_smoothed_nll_loss_transformers
from time import time
from pathlib import Path
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import get_linear_schedule_with_warmup, GenerationConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from evaluate import load


def fault_tolerant(func):
    def wrapper(*args, **kwargs):
        try:
            x = func(*args, **kwargs)
        except:
            return None
        return x

    return wrapper


class GradientsPrintingCallback(Callback):
    def on_after_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        for n, p in pl_module.named_parameters():
            pl_module.print(n, ":", p.grad)


class TeamsCallback(Callback):
    def __init__(self, hookurl: str) -> None:
        super().__init__()
        self.hookurl = hookurl

    @fault_tolerant
    @rank_zero_only
    def evaluation_msg(self, stage: str, pl_module):
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title(f"✔️ {str(pl_module.__class__.__name__)} {stage} End")

        rouge_scores = pl_module.scores[-1]
        if isinstance(rouge_scores, dict):
            section = pymsteams.cardsection()
            section.title("Rouge Scores")
            for k, v in rouge_scores.items():
                section.addFact(k, v)
            msg.addSection(section)

        msg.text(f"Check out {stage} summary :")
        msg.send()

    @fault_tolerant
    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title(f"🚀 {str(pl_module.__class__.__name__)} training started")
        msg.text("Fit loop begins.")
        msg.send()

    @fault_tolerant
    @rank_zero_only
    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        msg = pymsteams.connectorcard(hookurl=self.hookurl)
        msg.title("❌ Training exception")
        msg.text(f"An error occured : \n{exception}")
        msg.send()

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.evaluation_msg("validation", pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.evaluation_msg("test", pl_module)


class LongformerLightning(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> None:
        super().__init__()
        self.args = model_config
        self.generation_config = generation_config

        self.losses = []
        self.scores = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.retriever_generator = None
        self.rouge = load(
            self.args.rouge_path,
            experiment_id=self.args.mips_cache_prefix,
        )

        self.validation_outputs_path = Path(self.args.validation_outputs_dir)
        self.validation_outputs_path.mkdir(parents=True, exist_ok=True)

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
                self.retriever_generator.encoder.mips.encoder.requires_grad_(False)
                self.retriever_generator.encoder.query_encoder.requires_grad_(False)
            elif self.args.mips_encoder_freezed and not self.args.mips_disabled:
                self.retriever_generator.encoder.mips.encoder.requires_grad_(False)

    def on_train_batch_start(
        self, batch: dict, batch_idx: int, unused: int = 0
    ) -> None:
        if not self.args.mips_no_init_build and not self.args.mips_disabled:
            is_update_step = self.global_step % self.args.mips_rebuild_every == 0
            is_step_built = (
                self.global_step in self.retriever_generator.encoder.mips.rebuilt_steps
            )
            if (
                not self.args.mips_freezed
                and not self.args.mips_encoder_freezed
                and not is_step_built
                and is_update_step
            ):
                self._build_mips_index2()

    def on_fit_start(self) -> None:
        if not self.args.mips_no_init_build and not self.args.mips_disabled:
            self._build_mips_index2()

    def _build_mips_index2(self) -> None:
        self.retriever_generator.encoder.mips.init_embeddings_folder()
        self.trainer.strategy.barrier()
        self.retriever_generator.encoder.mips.encode_text2(
            rank=self.global_rank,
            num_rank=self.trainer.num_devices * self.trainer.num_nodes,
        )
        self.trainer.strategy.barrier()
        self.retriever_generator.encoder.mips.build_index()
        self.retriever_generator.encoder.mips.save()
        self.trainer.strategy.barrier()
        self.retriever_generator.encoder.mips.rebuilt_steps.append(self.global_step)
        self.retriever_generator.encoder.mips.load()

    def on_predict_start(self) -> None:
        self.on_fit_start()

    def on_test_start(self) -> None:
        self.on_fit_start()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        index: list,
        aid: list,
        aid_counts: torch.Tensor,
        target: list,
        input: list = None,
        **kwargs,
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
            target_str=target,
            input_str=input,
            return_dict=True,
            logger=self.log_dict,
        )

        decoder_input_ids_shifted = (
            self.retriever_generator.model.prepare_decoder_input_ids_from_labels(labels)
        )

        decoder_head_outputs = self.retriever_generator(
            input_ids=decoder_input_ids_shifted,
            attention_mask=None,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_attentions=self.args.output_copy_probs,
        )

        if not self.args.mips_disabled:
            lprobs = decoder_head_outputs.logits
        else:
            lprobs = torch.nn.functional.log_softmax(
                decoder_head_outputs.logits, dim=-1
            )

        if self.args.log_copy_metrics and not self.args.mips_disabled:
            k = 10
            copy_gate: torch.Tensor = decoder_head_outputs.copy_gate
            copy_probs: torch.Tensor = decoder_head_outputs.copy_probs

            memory_length = copy_probs.shape[2] // self.args.mips_topk

            all_max_copy_probs, all_index = copy_probs.max(2)
            topk_max_copy_probs, topk_index = all_max_copy_probs.topk(k)

            topk_index = torch.div(
                all_index.gather(1, topk_index), memory_length, rounding_mode="floor"
            )
            all_index = torch.div(all_index, memory_length, rounding_mode="floor")

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

        return lprobs

    def training_step(self, batch, batch_idx):
        self.train()
        if not self.args.mips_disabled:
            self.retriever_generator.encoder.query_encoder.train(
                not self.args.mips_freezed
            )
            self.retriever_generator.encoder.mips.encoder.train(
                not (self.args.mips_freezed or self.args.mips_encoder_freezed)
            )

        log_probs = self(**batch)

        loss = label_smoothed_nll_loss_transformers(
            log_probs=-log_probs.view(-1, len(self.retriever_generator.tokenizer)),
            labels=batch["labels"].view(-1),
            epsilon=self.args.label_smoothing_eps,
            ignore_index=self.retriever_generator.pad_token_id,
        )

        self.log(
            "loss",
            loss.item(),
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def generate(self, batch: dict):
        self.eval()

        encoder_outputs: RGEncoderModelOutput = self.retriever_generator.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            query_input_ids=batch["query_input_ids"],
            query_attention_mask=batch["query_attention_mask"],
        )
        kwargs = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": batch["attention_mask"],
        }

        output = self.retriever_generator.generate(
            generation_config=self.generation_config,
            pad_token_id=self.retriever_generator.tokenizer.pad_token_id,
            decoder_start_token_id=self.retriever_generator.config.decoder_start_token_id,
            bos_token_id=self.retriever_generator.config.bos_token_id,
            eos_token_id=self.retriever_generator.config.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=self.args.output_copy_probs,
            **kwargs,
        )

        return output, encoder_outputs

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        output, encoder_outputs = self.generate(batch)

        predictions = self.retriever_generator.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        references = batch["target"]

        tokens = [
            self.retriever_generator.tokenizer.convert_ids_to_tokens(seq)
            for seq in output.sequences
        ]

        tokens_sequence_copy_probs = output.get("cross_attentions", None)
        tokens_copy_probs = None
        if self.args.output_copy_probs:
            if hasattr(output, "beam_indices"):
                tokens_copy_probs = [
                    [
                        tokens_sequence_copy_probs[k][
                            j, :, output.sequences[i, k + 1]
                        ].item()
                        if j != -1
                        else 0.0
                        for k, j in enumerate(output.beam_indices[i])
                    ]
                    for i in range(output.beam_indices.shape[0])
                ]

        output = {
            "query": batch["query_input"],
            "predictions": predictions,
            "references": references,
            "examples": encoder_outputs.examples,
            "tokens": tokens,
            "tokens_copy_probs": tokens_copy_probs,
            # "tokens_sequence_copy_probs": tokens_sequence_copy_probs, # High memory consomption
        }

        return output

    def validation_step(self, batch, batch_idx) -> dict:
        output = self.predict_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        self.rouge.add_batch(
            predictions=output["predictions"],
            references=output["references"],
        )
        return output

    def on_validation_epoch_end(self) -> None:
        rouge_scores = self.rouge.compute()
        self.scores.append(rouge_scores)
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)

        filename = (
            self.validation_outputs_path / f"output-{self.trainer.current_epoch}.json"
        )
        with open(filename, mode="w") as f:
            json.dump(self.validation_step_outputs, f)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> dict:
        output = self.predict_step(batch, batch_idx)
        self.test_step_outputs.append(output)
        self.rouge.add_batch(
            predictions=output["predictions"],
            references=output["references"],
        )
        return output

    def on_test_epoch_end(self) -> tuple:
        rouge_scores = self.rouge.compute()
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)

        ckpt_name, ckpt_type = "", "ZeroShot"
        if isinstance(self.trainer.ckpt_path, (str, Path)):
            splited_path = self.trainer.ckpt_path.split("/")
            ckpt_name = splited_path[-1].replace(".ckpt", "")
            ckpt_type = splited_path[-2]
        args = f"num_beams={self.generation_config.num_beams}"

        filename = f"./{ckpt_type}-{ckpt_name}-{int(time())}-{args}-outputs.json"
        with open(filename, mode="w") as f:
            json.dump(self.test_step_outputs, f)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(), lr=self.args.lr
            )
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        # optimizer = FusedAdam(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
