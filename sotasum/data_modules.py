import pytorch_lightning as pl
import re
import datasets
import rich

from .bart_lightning import BartModelConfig
from .model_config import ModelConfig
from .t5_lightning import T5ModelConfig
from copy import deepcopy
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class MultiXScienceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_config: BartModelConfig = BartModelConfig(),
        path: str = "multi_x_science_sum",
        save_path: str = "./multixscience.arrow",
        doc_sep: str = "\n" * 2,
        aggregation: bool = False,
        num_workers: int = 0,
        max_length: int = 1024,
        num_proc_tokenization: int = None,
    ) -> None:
        super().__init__()
        self.args = model_config
        self.path = path
        self.save_path = save_path
        self.doc_sep = doc_sep
        self.aggregation = aggregation
        self.num_workers = num_workers
        self.max_length = max_length
        self.num_proc_tokenization = num_proc_tokenization

    def setup(self, stage: str) -> None:
        self.data = datasets.load_from_disk(
            dataset_path=self.save_path,
        )

    def prepare_data(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.pretrained_model_name_or_path,
        )
        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_length,
        }

        data = datasets.load_dataset(
            path=self.path,
        )
        if self.aggregation:
            data = (
                data.to_pandas()
                .groupby("aid")
                .agg(
                    {
                        "mid": lambda x: x.to_list()[0],
                        "abstract": lambda x: x.to_list()[0],
                        "ref_abstract": lambda x: {
                            k: [j for i in x.to_list() for j in i[k] if bool(j)]
                            for k in x.to_list()[0].keys()
                        },
                        "related_work": lambda x: " ".join(x.to_list()),
                    }
                )
            )
            data = datasets.Dataset.from_pandas(data, preserve_index=True)

        data = data.map(
            self._prepare_data,
            desc="Preparing data",
        )
        data = data.map(
            lambda x: tokenizer(
                text=x["input"], text_target=x["target"], **tokenizer_kwargs
            ),
            batched=True,
            num_proc=self.num_proc_tokenization,
            desc="Tokenization",
        )
        data.save_to_disk(self.save_path)

    def _prepare_data(self, x) -> dict:
        _input = [x["abstract"]]
        _input += [a for a in x["ref_abstract"]["abstract"] if a != ""]
        _input = self.doc_sep.join(_input)

        references = re.sub(r"\@cite_\d+", "cite", x["related_work"]).strip()

        output = {
            "input": _input,
            "target": references,
        }
        return output

    def _get_dataloader(self, mode: str, batch_size: int) -> DataLoader:
        dataset = self.data[mode].select_columns(
            column_names=["input_ids", "attention_mask", "labels", "input", "target"],
        )
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return data_loader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train", self.args.batch_size)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test", self.args.validation_batch_size)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("validation", self.args.validation_batch_size)


class PromptMultiXScienceDataModule(MultiXScienceDataModule):
    def __init__(
        self,
        model_config: T5ModelConfig = T5ModelConfig(),
        path: str = "multi_x_science_sum",
        save_path: str = "./multixscience.arrow",
        doc_sep: str = "\n" * 2,
        aggregation: bool = False,
        num_workers: int = 0,
        max_length: int = 1024,
        num_proc_tokenization: int = None,
        prompt: str = "summarize: ",
    ) -> None:
        super().__init__(
            model_config,
            path,
            save_path,
            doc_sep,
            aggregation,
            num_workers,
            max_length,
            num_proc_tokenization,
        )
        self.prompt = prompt

    def _prepare_data(self, x) -> dict:
        output = super()._prepare_data(x)
        output["input"] = f"{self.prompt}{output['input']}"
        return output


class RGMultiXScienceDataModule(MultiXScienceDataModule):
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        path: str = "multi_x_science_sum",
        save_path: str = "./multixscience.arrow",
        aggregation: bool = False,
        num_workers: int = 0,
        max_length: int = 1024,
        query_max_length: int = 1024,
        decoder_max_length: int = 1024,
        num_proc_tokenization: int = None,
    ) -> None:
        super().__init__()
        self.args = model_config
        self.path = path
        self.save_path = save_path
        self.aggregation = aggregation
        self.num_workers = num_workers
        self.max_length = max_length
        self.query_max_length = query_max_length
        self.decoder_max_length = decoder_max_length
        self.num_proc_tokenization = num_proc_tokenization

    def setup(self, stage: str) -> None:
        self.data = datasets.load_from_disk(
            dataset_path=self.save_path,
        )

    def prepare_data(self) -> None:
        data = datasets.load_dataset(
            path=self.path,
        )

        data_dict = {}
        for k in data.keys():
            v = data[k].to_pandas()
            if self.aggregation:
                v = v.groupby("aid", as_index=False).agg(
                    {
                        "mid": lambda x: x.to_list()[0],
                        "abstract": lambda x: x.to_list()[0],
                        "ref_abstract": lambda x: {
                            k: [j for i in x.to_list() for j in i[k] if bool(j)]
                            for k in x.to_list()[0].keys()
                        },
                        "related_work": lambda x: " ".join(x.to_list()),
                    }
                )
            v = v.merge(
                v["aid"].value_counts(),
                right_index=True,
                left_on="aid",
            ).rename(columns={"count": "aid_counts"})
            data_dict.update(
                {k: datasets.Dataset.from_pandas(v, split=k, preserve_index=True)}
            )
        data = datasets.DatasetDict(data_dict)
        data = data.rename_column("__index_level_0__", "index")

        data = data.map(
            self._prepare_data,
            desc="Preparing data",
        )

        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.model_name,
        )
        _ = tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.args.doc_sep]}
        )

        query_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.query_encoder_path,
        )

        def tokenize(x) -> dict:
            query = query_tokenizer(
                text=x["query_input"],
                max_length=self.query_max_length,
                **tokenizer_kwargs,
            )
            query = {f"query_{k}": v for k, v in query.items()}

            _input = tokenizer(
                text=x["input"], max_length=self.max_length, **tokenizer_kwargs
            )
            _input = {k: v for k, v in _input.items()}

            output = {**_input, **query}
            output["labels"] = tokenizer(
                text=x["target"],
                max_length=self.decoder_max_length,
                return_attention_mask=False,
                **tokenizer_kwargs,
            )["input_ids"]

            return output

        data = data.map(
            tokenize,
            batched=True,
            num_proc=self.num_proc_tokenization,
            desc="Tokenization",
        )

        data.save_to_disk(self.save_path)

    def _prepare_data(self, x) -> dict:
        _input = [x["abstract"]]
        if self.args.source_memory:
            _input += [a for a in x["ref_abstract"]["abstract"] if a != ""]
        _input = self.args.doc_sep.join(_input)

        query_input = deepcopy(x["abstract"])

        references = re.sub(r"\@cite_\d+", "cite", x["related_work"]).strip()

        output = {
            "input": _input,
            "query_input": query_input,
            "target": references,
        }
        return output

    def _get_dataloader(self, mode: str, batch_size: int) -> DataLoader:
        dataset = self.data[mode].select_columns(
            column_names=[
                "input_ids",
                "attention_mask",
                "query_input_ids",
                "query_attention_mask",
                "labels",
                "input",
                "query_input",
                "target",
                "aid",
                "aid_counts",
                "index",
            ],
        )
        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "query_input_ids",
                "query_attention_mask",
                "labels",
            ],
            output_all_columns=True,
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return data_loader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train", self.args.batch_size)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test", self.args.validation_batch_size)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("validation", self.args.validation_batch_size)
