import logging
import rich

from datasets import disable_caching
from lightning_model import LongformerLightning, GradientsPrintingCallback, TeamsCallback
from retriever_generator import RetrieverGenerator
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, RichModelSummary, RichProgressBar
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from rich.traceback import install


class CustomCLI(LightningCLI):

    def before_fit(self):
        for logger in self.trainer.loggers:
            logger.log_hyperparams(self.config)


def cli():
    CustomCLI(
        model_class=LongformerLightning,
        save_config_kwargs={"overwrite": True},
    )

# def train(args: argparse.Namespace) -> None:
#     tb_logger = TensorBoardLogger(
#         save_dir=args.tensorboard_save_dir,
#         log_graph=True,
#     )

#     logger = [tb_logger]

#     filename = '{epoch}-{rouge1:.2f}'
#     if not args.disable_mlflow:
#         mlflow_logger = MLFlowLogger(
#             experiment_name=f"{args.mlflow_exp_prefix}_sotasum",
#             tracking_uri=f"file:{args.mlflow_mlruns_dir}/mlruns",
#         )
#         logger.append(mlflow_logger)
#         mlflow_run = mlflow_logger.experiment.get_run(mlflow_logger.run_id)
#         # filename += f"-{mlflow_run.info.run_name}"

#     args.seed = pl.seed_everything(args.seed, workers=True)
#     rich.print(str(args).replace(", ", ",\n"))

#     checkpoint_callback = ModelCheckpoint(
#         dirpath=args.checkpoint_dir,
#         save_top_k=1,
#         monitor="rouge1",
#         mode="max",
#         filename=filename,
#     )

#     progress_bar_callback = TQDMProgressBar(
#         refresh_rate=args.pb_refresh_rate *
#         args.accumulate_grad_batches if args.accumulate_grad_batches is not None else 1,
#     )

#     strategy = None
#     if args.deepspeed:
#         ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN
#         strategy = DeepSpeedStrategy(
#             stage=2,
#             offload_optimizer=True,
#             # offload_parameters=True,
#             logging_level=ds_logging_level,
#             initial_scale_power=4,
#             # allgather_bucket_size=5e8,
#             # reduce_bucket_size=5e8,
#         )
#     else:
#         pass
#         # strategy = "fsdp_native"
#         # strategy = DDPShardedStrategy()
#         # strategy = DDPStrategy(
#         #     static_graph=True,
#         # )

#     profiler = None
#     if args.enable_profiler:
#         profiler = PyTorchProfiler(
#             # dirpath='./profile/',
#             filename="sotasum",
#             profile_memory=True,
#         )

#     callbacks = [
#         checkpoint_callback,
#         progress_bar_callback,
#     ]

#     if args.print_gradients:
#         callbacks.append(GradientsPrintingCallback())

#     if args.teams_hookurl is not None:
#         callbacks.append(TeamsCallback(args.teams_hookurl))

#     trainer: pl.Trainer = pl.Trainer.from_argparse_args(
#         args,
#         profiler=profiler,
#         logger=logger,
#         callbacks=callbacks,
#         strategy=strategy,
#         log_every_n_steps=4,
#     )

#     rg_model = RetrieverGenerator(args=args)
#     pl_model = LongformerLightning(args=args, model=rg_model)

#     trainer.fit(
#         model=pl_model,
#         ckpt_path=args.checkpoint_path,
#     )

#     return trainer


# def test(args: argparse.Namespace):

#     tb_logger = TensorBoardLogger(
#         save_dir=f"{args.tensorboard_save_dir}_test",
#     )

#     logger = [tb_logger]

#     strategy = None
#     if args.deepspeed:
#         ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN
#         strategy = DeepSpeedStrategy(
#             stage=2,
#             offload_optimizer=True,
#             # offload_parameters=True,
#             logging_level=ds_logging_level,
#             initial_scale_power=4,
#             # allgather_bucket_size=5e8,
#             # reduce_bucket_size=5e8,
#         )

#     pl_model = LongformerLightning(
#         args=args,
#         model=RetrieverGenerator(args=args),
#     )

#     trainer: pl.Trainer = pl.Trainer.from_argparse_args(
#         args=args,
#         strategy=strategy,
#         logger=logger
#     )

#     trainer.test(
#         model=pl_model,
#         ckpt_path=args.checkpoint_path,
#     )


# def predict(args: argparse.Namespace):
#     ds_logging_level = logging.DEBUG if args.deepspeed_log else logging.WARN

#     strategy = DeepSpeedStrategy(
#         stage=2,
#         offload_optimizer=True,
#         # offload_parameters=True,
#         logging_level=ds_logging_level,
#         initial_scale_power=4,
#         # allgather_bucket_size=5e8,
#         # reduce_bucket_size=5e8,
#     )

#     pl_model = LongformerLightning(
#         args=args,
#         model=RetrieverGenerator(args=args),
#     )

#     test_dataloader = pl_model._get_data_loader(
#         "test",
#         batch_size=args.batch_size,
#         select_indices=range(4),
#     )

#     trainer: pl.Trainer = pl.Trainer.from_argparse_args(
#         args=args,
#         strategy=strategy,
#     )

#     outputs = trainer.predict(
#         model=pl_model,
#         dataloaders=test_dataloader,
#         ckpt_path=args.checkpoint_path,
#     )

#     console = Console(record=True)
#     colors = ["#33FF00", "#33FF33", "#33FF66", "#33FF99", "#33FFCC", "#33FFFF"]

#     texts = []
#     for output in outputs:
#         for tokens, copy_prob in zip(output['tokens'], output['tokens_copy_probs']):
#             text = Text()
#             for token, prob in zip(tokens, copy_prob):
#                 color_id = int(prob // (1/len(colors)))
#                 text.append(f"{token}/{prob:0.2f} ", style=colors[color_id])
#             texts.append(text)

#     for t in texts:
#         print()
#         console.print(t)

#     # with open("./test.json", mode="w") as f:
#     #     json.dump(outputs, f)


if __name__ == "__main__":
    # install(show_locals=False)
    disable_caching()
    hf_logger = logging.getLogger('transformers')
    hf_logger.setLevel(logging.WARN)
    cli()
