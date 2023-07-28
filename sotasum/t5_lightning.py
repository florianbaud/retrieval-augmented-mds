import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import torch
import rich

from dataclasses import dataclass
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig, PreTrainedTokenizer
from evaluate.utils.file_utils import DownloadConfig
from transformers import get_linear_schedule_with_warmup
from evaluate import load as load_evaluate
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers.models.bart.modeling_bart import shift_tokens_right


def tensortolist(t):
    if isinstance(t, (list, tuple)):
        return [tensortolist(e) for e in t]
    elif isinstance(t, torch.Tensor):
        return t.tolist()
    return t


def fault_tolerant(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as error:
            rich.print(error)
        return None
    return wrapper


@dataclass
class T5ModelConfig:
    pretrained_model_name_or_path: str = "t5-large"
    learning_rate: float = 1e-03
    rouge_path: str = "rouge"
    gradient_checkpointing: bool = False
    batch_size: int = 1
    validation_batch_size: int = 2
    label_smoothing: float = 0.1


class T5Lightning(pl.LightningModule):

    def __init__(
        self,
        model_config: T5ModelConfig,
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.generation_config = generation_config

        self.model = None
        self.tokenizer = None
        self.rouge = None
        self.mlflow_logger = None

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.scores = []

    def setup(self, stage: str) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
        )
        if self.model_config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                rich.print(
                    "gradient_checkpointing is True but model has no `gradient_checkpointing_enable` method !")

        self.tokenizer: PreTrainedTokenizer = T5Tokenizer.from_pretrained(
            self.model_config.pretrained_model_name_or_path,
        )

        download_config = DownloadConfig(local_files_only=True)
        self.rouge = load_evaluate(
            path=self.model_config.rouge_path,
            download_config=download_config,
            # experiment_id=self.args.mips_cache_prefix,
        )

        for logger in self.loggers:
            if isinstance(logger, MLFlowLogger):
                self.mlflow_logger = logger

    def forward(self, batch, use_cache: bool = True):
        decoder_input_ids = self.model._shift_right(batch['labels'])
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            use_cache=use_cache,
        )
        return output

    def training_step(self, batch, batch_idx):
        self.train()
        output = self(batch, use_cache=False)
        loss = F.cross_entropy(
            output.logits.view(-1, output.logits.size(-1)),
            batch['labels'].view(-1),
            label_smoothing=self.model_config.label_smoothing,
            ignore_index=self.model.config.pad_token_id,
        )
        self.log(
            'loss',
            loss.item(),
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        self.eval()

        output = self.model.generate(
            inputs=batch['input_ids'],
            generation_config=self.generation_config,
            attention_mask=batch['attention_mask'],
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        predictions = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True)
        # predictions = [p.split("### Response:")[-1] for p in full_predictions]

        output = dict(output)
        output.update({
            "predictions": predictions,
            # "predictions": predictions,
            "references": batch['target'],
        })
        # dict_keys(['sequences', 'scores', 'predictions', 'references'])

        _ = output.pop("scores", None)
        output = {k: tensortolist(v) for k, v in output.items()}

        return output

    def validation_step(self, batch, batch_idx) -> None:
        output = self.predict_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        self.rouge.add_batch(
            predictions=output['predictions'],
            references=output['references'],
        )

    def test_step(self, batch, batch_idx) -> None:
        output = self.predict_step(batch, batch_idx)

        self.test_progress += len(output['predictions'])
        if self.local_rank == 0:
            self.logger.log_metrics({"progress": 1.0}, step=self.test_progress)

        self.test_step_outputs.append(output)
        self.rouge.add_batch(
            predictions=output['predictions'],
            references=output['references'],
        )

    def on_validation_start(self) -> None:
        self.validation_step_outputs.clear()

    def on_test_start(self) -> None:
        self.test_step_outputs.clear()
        self.test_progress = 0

    def on_validation_epoch_end(self) -> None:
        rouge_scores = self.rouge.compute()
        self.scores.append(rouge_scores)
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)
        self._log_outputs(self.validation_step_outputs, 'validation')

    def on_test_epoch_end(self) -> None:
        rouge_scores = self.rouge.compute()
        self.scores.append(rouge_scores)
        for k, v in rouge_scores.items():
            self.log(k, v, sync_dist=True)
        self._log_outputs(self.test_step_outputs, 'test')

    @fault_tolerant
    def _log_outputs(self, outputs: list, stage: str) -> None:
        file_name = f"{self.global_step}/{stage}_outputs_{self.local_rank}.json"
        sel = ['predictions', 'references']

        df = pd.concat(pd.DataFrame(
            {k: v for k, v in o.items() if k in sel}) for o in outputs)
        outputs_dict = df.to_dict(orient="records")

        # sequences = {"sequences": [str(o['sequences']) for o in outputs]}
        # rich.print(sequences['sequences'])

        if isinstance(self.mlflow_logger, MLFlowLogger):
            self.mlflow_logger._mlflow_client.log_dict(
                run_id=self.mlflow_logger.run_id,
                dictionary=outputs_dict,
                artifact_file=file_name,
            )
            # self.mlflow_logger._mlflow_client.log_dict(
            #     run_id=self.mlflow_logger.run_id,
            #     dictionary=sequences,
            #     artifact_file=file_name,
            # )

    def configure_optimizers(self):

        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = DeepSpeedCPUAdam(
                model_params=self.parameters(),
                lr=self.model_config.learning_rate,
            )
        else:
            optimizer = torch.optim.AdamW(
                params=self.trainer.model.parameters(),
                lr=self.model_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=0.01,
            )

        return optimizer
