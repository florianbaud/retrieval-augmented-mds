import torch
import logging
import argparse
import rich
import pytorch_lightning as pl

from rich.logging import RichHandler
from lightning_model import LongformerLightning, GradientsPrintingCallback, TeamsCallback
from retriever_generator import RetrieverGenerator
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, RichModelSummary, RichProgressBar
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from rich.traceback import install


def get_args(args: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # Script args
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pb-refresh-rate", type=int, default=1)
    parser.add_argument("--tensorboard-save-dir", type=str, default='tb_logs')
    parser.add_argument("--disable-mlflow", action="store_true", default=False)
    parser.add_argument("--mlflow-exp-prefix", type=str, default="")
    parser.add_argument("--mlflow-mlruns-dir", type=str, default=".")
    parser.add_argument("--enable-profiler",
                        action="store_true", default=False)
    parser.add_argument("--print-gradients",
                        action="store_true", default=False)
    parser.add_argument("--deepspeed-log", action="store_true", default=False)
    parser.add_argument("--teams-hookurl", type=str, default=None)

    # Validation args
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--validation-outputs-dir",
                        type=str, default='./outputs/')
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--use-cache", action="store_true", default=False)
    parser.add_argument("--rouge-path", type=str, default='rouge')
    parser.add_argument("--generate-max-length", type=int, default=1024)

    # Training args
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    # LongFormer pretraining settings : polynomial decay, lr=3e-5, warmup=500
    # PRIMERA pretraining settings : linear decay, lr=3e-5, warmup=10 000, total-step=100 000

    # Models args
    parser.add_argument("--model-name", type=str,
                        help="Name of pretrained model.")
    parser.add_argument("--copy-decoder-layers", type=int, default=8)
    parser.add_argument("--model-cache-dir", type=str, help="Model cache dir.")
    parser.add_argument("--join-method", type=str,
                        default="concat_start_wdoc_global")
    parser.add_argument("--attention-mode", type=str, default="sliding_chunks")
    parser.add_argument("--query-encoder-path", type=str,
                        default="allenai/longformer-large-4096")
    parser.add_argument("--query-state-dict", type=str, default=None)
    parser.add_argument("--model-tok-max-length", type=int, default=None)
    parser.add_argument("--decoder-max-length", type=int, default=None)
    parser.add_argument("--query-tok-max-length", type=int, default=None)
    parser.add_argument("--label-smoothing-eps", type=float, default=0.1)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--use-own-decoder",
                        action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing",
                        action="store_true", default=False)
    parser.add_argument("--output-copy-probs",
                        action="store_true", default=False)
    parser.add_argument("--gates-mode", default="nmt",
                        choices=['nmt', 'onlycopy', 'both'])
    parser.add_argument("--skip-residual", action="store_true", default=False)
    parser.add_argument(
        "--memory-forcing", choices=['target_only', 'target_in', "no_forcing"], default="no_forcing")

    # Memory args
    parser.add_argument("--memory-model-name", type=str,
                        default="allenai/longformer-large-4096")
    parser.add_argument("--memory-tok-max-length", type=int, default=None)

    # MIPS args
    parser.add_argument("--mips-disabled", action="store_true", default=False)
    parser.add_argument("--mips-freezed", action="store_true", default=False)
    parser.add_argument("--mips-batch-size", type=int, default=32)
    parser.add_argument("--mips-num-gpus", type=int, default=0)
    parser.add_argument("--mips-topk", type=int, default=2)
    parser.add_argument("--mips-string-factory",
                        type=str, default="IVF256,SQ8")
    parser.add_argument("--mips-rebuild-every", type=int, default=10000)
    parser.add_argument("--mips-train-size", type=int, default=None)
    parser.add_argument("--mips-metric-type", type=int, default=0,
                        help="Choose between : 0 -> INNER_PRODUCT ; 1 -> L2")
    # parser.add_argument("--mips-normalize", type=bool, default=True)
    parser.add_argument("--mips-no-normalize",
                        action="store_false", default=True)
    parser.add_argument("--mips-dataset", type=str, default="multi_x_science")
    parser.add_argument("--mips-data-script-path", type=str,
                        default="multi_x_science_sum")
    parser.add_argument("--mips-model-name", type=str,
                        default="allenai/longformer-large-4096")
    parser.add_argument("--mips-state-dict", type=str, default=None)
    # parser.add_argument("--mips-save-path", type=str, default=None)
    # parser.add_argument("--mips-load-path", type=str, default=None)
    parser.add_argument("--mips-no-init-build",
                        action="store_true", default=False)
    parser.add_argument("--mips-db-max-size", type=int, default=None)
    parser.add_argument("--mips-tok-max-length", type=int, default=None)
    parser.add_argument("--mips-tmp-max-norm-file",
                        type=str, default="max_norm.pkl")
    parser.add_argument("--mips-tmp-index-file",
                        type=str, default="index.faiss")
    parser.add_argument("--mips-tmp-embeddings-folder",
                        type=str, default="embeddings")
    parser.add_argument("--mips-tmp-folder", type=str, default='./tmp')
    parser.add_argument("--mips-cache-prefix", type=str, default="")

    # Data args
    parser.add_argument("--dataset-name", type=str, default="multi_x_science")
    parser.add_argument("--data-script-path", type=str,
                        default="multi_x_science_sum")
    parser.add_argument("--data-path", type=str, default='../data_hf')
    parser.add_argument("--doc-sep", type=str, default="<DOC_SEP>")
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument("--clean-cache", action="store_true", default=False)
    parser.add_argument("--copy-forcing", type=float, default=0.0)

    args = parser.parse_args(args)
    return args


def train(args: argparse.Namespace) -> None:
    rich.print(*[f"{k} = {v}" for k, v in vars(args).items()], sep='\n')

    tb_logger = TensorBoardLogger(
        save_dir=args.tensorboard_save_dir,
        log_graph=True,
    )

    logger = [tb_logger]

    if not args.disable_mlflow:
        mlflow_logger = MLFlowLogger(
            experiment_name=f"{args.mlflow_exp_prefix}_sotasum",
            tracking_uri=f"file:{args.mlflow_mlruns_dir}/mlruns",
        )
        logger.append(mlflow_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        save_top_k=1,
        monitor="rouge1",
        mode="max",
        filename='{epoch}-{rouge1:.2f}',
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=args.pb_refresh_rate *
        args.accumulate_grad_batches if args.accumulate_grad_batches is not None else 1,
    )

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

    profiler = None
    if args.enable_profiler:
        profiler = PyTorchProfiler(
            # dirpath='./profile/',
            filename="sotasum",
            profile_memory=True,
        )

    callbacks = [
        checkpoint_callback,
        progress_bar_callback,
    ]

    if args.print_gradients:
        callbacks.append(GradientsPrintingCallback())

    if args.teams_hookurl is not None:
        callbacks.append(TeamsCallback(args.teams_hookurl))

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        strategy=strategy,
        log_every_n_steps=4,
    )

    rg_model = RetrieverGenerator(args=args)
    pl_model = LongformerLightning(args=args, model=rg_model)

    trainer.fit(
        model=pl_model,
        ckpt_path=args.checkpoint_path,
    )

    return trainer


def test(args: argparse.Namespace):

    tb_logger = TensorBoardLogger(
        save_dir=f"{args.tensorboard_save_dir}_test",
    )

    logger = [tb_logger]

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

    pl_model = LongformerLightning(
        args=args,
        model=RetrieverGenerator(args=args),
    )

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args=args,
        strategy=strategy,
        logger=logger
    )

    with torch.inference_mode():
        trainer.test(
            model=pl_model,
            ckpt_path=args.checkpoint_path,
        )


def predict(args: argparse.Namespace):
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

    pl_model = LongformerLightning(
        args=args,
        model=RetrieverGenerator(args=args),
    )

    test_dataloader = pl_model._get_data_loader(
        "test",
        batch_size=args.batch_size,
        select_indices=range(4),
    )

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args=args,
        strategy=strategy,
    )

    with torch.inference_mode():
        outputs = trainer.predict(
            model=pl_model,
            dataloaders=test_dataloader,
            ckpt_path=args.checkpoint_path,
        )

    dict_outputs = [
        {"predictions": p, "target": r, "examples": e} for o in outputs for p, r, e in zip(o[0], o[1], o[2])
    ]

    rich.print(dict_outputs)


if __name__ == "__main__":
    install(show_locals=False)
    args = get_args()

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=10,
        format=FORMAT,
        datefmt="[%X]",
        # handlers=[RichHandler()],
    )

    hf_logger = logging.getLogger('transformers')
    hf_logger.setLevel(logging.ERROR)

    # if args.mips_num_gpus > 0:
    #     args.mips_num_gpus = min(args.mips_num_gpus, torch.cuda.device_count())

    # if args.seed is not None:
    args.seed = pl.seed_everything(args.seed, workers=True)

    if args.mode == 'train':
        trainer = train(args=args)
    elif args.mode == 'test':
        test(args=args)
    elif args.mode == 'predict':
        predict(args=args)
    else:
        assert False, "Unknown mode."
