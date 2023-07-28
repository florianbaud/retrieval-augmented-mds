import logging

from datasets import disable_caching
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


class CustomCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.init_args.model_config",
                              "data.init_args.model_config")

    def before_fit(self) -> None:
        for logger in self.trainer.loggers:
            logger.log_hyperparams(self.config)


def cli() -> None:
    CustomCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    # install(show_locals=False)
    disable_caching()
    hf_logger = logging.getLogger('transformers')
    hf_logger.setLevel(logging.WARN)
    cli()
